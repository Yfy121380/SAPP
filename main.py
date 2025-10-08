import datetime
import datasets
from datasets import load_dataset

from utils import *
import torch
import torch.nn as nn
from logger import get_logger
from utils import *
from training import Trainer
from transformers import BertModel, AutoConfig
import transformers
from gpt import *

transformers.logging.set_verbosity_error()
from args import *
from path_config import *

from attack import *

# 配置项
parser = get_parser()
args = parser.parse_args()

# # 结果保存
# wandb.init(project="prompt_dp", config=args)
logger = get_logger(
    log_file=f"{args.embedding_type}_{args.mapping_strategy}_{args.privatization_strategy}_eps_{args.eps}_top_{args.top_k}_save_{args.save_stop_words}_{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.txt")
logger.info(f"{args.dataset}, args: {args}")

if __name__ == "__main__":
    parser = get_parser()
    config_path = PathConfig()
    train_config_path = trainPathConfig()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # train_data, test_data = load_data(args.dataset)

    data = load_dataset(path="parquet", data_dir="./datasets/rotten_tomatoes")
    train_data = data["train"]
    test_data = data["validation"]
    origin_test_data = data["validation"]

    train_data = train_data.to_pandas()
    test_data = test_data.to_pandas()
    origin_test_data = origin_test_data.to_pandas()
    train_data.columns = ["sentence", "labels"]
    test_data.columns = ["sentence", "labels"]
    origin_test_data.columns = ["sentence", "labels"]

    #sim_word_dict, p_dict = get_customized_mapping(eps=args.eps, top_k=args.top_k)

    # with open(f"p_dict/eps_1.0_top_20.txt",
    #           'r') as dic:
    #     p_dict = json.load(dic)
    #
    # with open(f"sim_word_dict/eps_1.0_top_20.txt",
    #           'r') as dic:
    #     sim_word_dict = json.load(dic)

    #test_data = generate_new_sentences(df=test_data, sim_word_dict=sim_word_dict, p_dict=p_dict,
    #                                   question_path=config_path.question_candidates,
    #                                   sentence_path=config_path.sentence_candidates,
    #                                   sentence_number=args.s_number)
    # 计算每一个分句的困惑度指标
    ppl_calculate(config_path.sentence_candidates, config_path.sentence_ppl)

    # # 根据ppl指标进行候选句子筛选
    filter_ppl_sentence(config_path.sentence_ppl, config_path.sentence_ppl_filter,
                        final_sentence_number=args.final_number)

    # # 根据困惑度筛选过后的候选句子计算simcse相似度
    simcse_similarity(config_path.sentence_ppl_filter, config_path.sentence_simcse)
    # simcse_similarity_original_sentence(config_path.sentence_simcse)
    # delete_original_sentence(config_path.sentence_simcse)

    # 利用指数机制生成候选概率
    calculate_probabilities(config_path.sentence_simcse, config_path.sentence_probability, args.eps)

    sort_json_by_key(config_path.sentence_probability)

    # # # # 按照候选概率随机生成最终替换
    test_data = replace_with_probability(config_path.sentence_probability, test_data, "sentence")


    # # 按照候选概率随机生成最终替换
    bert_embedding_attack(origin_test_data, test_data)
    k1, k2 = knn_attack(origin_test_data, test_data)
    print(f"隐私预算为{args.eps}的KNN攻击成功率为：{k2 / k1}")

    calculate_classify_accuracy(test_data)

    # calculate_F1(test_data,origin_test_data, args.eps)

    # train_dataset = Bert_dataset(train_data)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_dataset = Bert_dataset(test_data)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    # logger.info(f"train_data:{len(train_data)},test_data:{len(test_data)}")
    #
    # model = BertForSequenceClassification.from_pretrained(
    #     args.model_type,
    #     cache_dir='./cache',
    #     num_labels=2,
    #     output_attentions=False,
    #     output_hidden_states=False)
    #
    # optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    #
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=0,
    #                                             num_training_steps=len(train_loader) * args.epochs)
    #
    # trainer = Trainer(
    #     model,
    #     scheduler,
    #     optimizer,
    #     args.epochs,
    #     args.log_steps,
    #     args.eval_steps,
    #     args.use_cuda,
    #     logger
    # )
    # trainer.train(train_loader, test_loader)
    #
    # acc = trainer.predict(test_loader)
    # logger.info(f"test acc = {acc:.4f}.")
