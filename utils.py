import os
import random
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch
import multiprocessing
import json
import os
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm, trange
from collections import Counter, defaultdict
import json
import string
import nltk
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch.nn.functional as F
import re
from scipy.spatial.distance import cosine
from torch.nn.functional import cosine_similarity

import warnings
from args import *
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, \
    AutoModel
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

import datetime
from logger import get_logger
from datasets import load_dataset

warnings.filterwarnings('ignore')

parser = get_parser()
args = parser.parse_args()


def sort_json_by_key(input_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        sorted_keys = sorted(data.keys(), key=lambda x: int(x))
        sorted_data = {key: data[key] for key in sorted_keys}

        with open(input_file, 'r', encoding='utf-8') as file1:
            json.dump(sorted_data, file1, indent=4)

        print("完成排序")
    except Exception as e:
        print(e)


def replace_with_probability(json_path, df, target_name):
    try:
        # 读取JSON数据
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取所有候选句子及其概率
        all_candidates = []
        for sent_id, sent_data in tqdm(data.items()):
            str1 = ""
            for outer_key, clauses in sent_data.items():
                for clause_key, details in clauses.items():
                    # 从 candidate_probabilities 中提取句子和概率

                    if "candidate_probabilities" in details:
                        candidates = list(details["candidate_probabilities"].keys())
                        probabilities = list(details["candidate_probabilities"].values())

                        selected_sentence = np.random.choice(
                            candidates,
                            1,
                            p=probabilities
                        )[0]
                        str1 = str1 + selected_sentence
                str1 = str1.strip()
            all_candidates.append(str1)

        if target_name == "question":
            for i in tqdm(range(len(df))):
                df.at[i, "question"] = all_candidates[i]

        if target_name == "sentence":
            for i in tqdm(range(len(df))):
                df.at[i, "sentence"] = all_candidates[i]

        return df

    except Exception as e:
        print(f"处理失败: {str(e)}")
        return df


def calculate_exponential_probabilities(similarities, eps):
    """
    为相似度数组计算指数机制概率
    :param similarities: 相似度数值列表
    :param eps: 隐私参数 ε
    :return: 概率数组
    """
    # 计算指数变换
    exp_values = np.exp(eps * np.array(similarities) / 2)
    # 归一化到概率分布
    probabilities = exp_values / np.sum(exp_values)
    return probabilities

def delete_original_sentence(input_file):

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功读取{len(data)}条JSON数据")

    for sent_id, sent_data in tqdm(data.items()):
        for original_sentence, layer2 in sent_data.items():
            for original_single_sentence, layer3 in layer2.items():
                candidate_dict = layer3.get("candidate_simcse", {})
                candidate_dict.pop()

    try:
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")
    return None

def calculate_probabilities(input_path, output_path, eps):
    """
    处理JSON文件，为每个candidate_simcse中的句子计算指数机制概率
    """
    try:
        # 读取JSON文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取 {len(data)} 条数据")

        # 遍历数据结构，处理candidate_simcse字段
        processed_count = 0
        for sent_id, sent_data in tqdm(data.items()):
            for original_sentence, clauses in sent_data.items():
                for clause, details in clauses.items():
                    # 检查是否存在candidate_simcse字段
                    if "candidate_simcse" in details:
                        simcse_dict = details["candidate_simcse"]
                        if simcse_dict:  # 确保字典不为空
                            # 提取相似度值
                            sentences = list(simcse_dict.keys())
                            similarities = list(simcse_dict.values())

                            # 计算指数概率
                            probabilities = calculate_exponential_probabilities(similarities, eps)

                            # 创建新的概率字典
                            prob_dict = {sent: prob for sent, prob in zip(sentences, probabilities)}

                            # 添加到原始数据中
                            details["candidate_probabilities"] = prob_dict
                            processed_count += 1
                        details.pop("candidate_simcse")

        print(f"成功处理 {processed_count} 个 candidate_simcse 字段")

        # 保存处理后的结果

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"结果已保存至 {output_path}")

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")


def simcse_similarity(input_file, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased", cache_dir="./cache2",
                                              local_files_only=True)
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased", cache_dir="./cache2",
                                      local_files_only=True).to(device)

    def calculate_simcse(original_text, text):
        texts = []
        texts.append(original_text)
        texts.append(text)
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Get the embeddings
        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

        # cosine_sim = 1 - cosine(embeddings[0], embeddings[1])

        cosine_sim = cosine_similarity(embeddings[0], embeddings[1], dim=0).item()
        return cosine_sim

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取{len(data)}条JSON数据")
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return

    new_results = {}

    for sent_id, sent_data in tqdm(data.items()):
        results = {}
        for original_sentence, layer2 in sent_data.items():
            results[original_sentence] = {}

            for original_single_sentence, layer3 in layer2.items():
                results[original_sentence][original_single_sentence] = {}

                valid_candidates = layer3.get("valid_candidates", 0)
                candidate_ppl_dict = layer3.get("candidate_ppl", {})

                candidate_simcse = {}
                for candidate, _ in candidate_ppl_dict.items():
                    score = calculate_simcse(original_single_sentence, candidate)
                    if score is not None:
                        candidate_simcse[candidate] = score

                results[original_sentence][original_single_sentence] = {
                    "valid_candidates": valid_candidates,
                    "candidate_simcse": candidate_simcse
                }
        new_results[sent_id] = results

    print(f"成功处理{len(new_results)}条数据")

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")

def simcse_similarity_original_sentence(input_file):
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased", cache_dir="./cache2",
                                              local_files_only=True)
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased", cache_dir="./cache2",
                                      local_files_only=True).to(device)

    def calculate_simcse(original_text, text):
        texts = []
        texts.append(original_text)
        texts.append(text)
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Get the embeddings
        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

        # cosine_sim = 1 - cosine(embeddings[0], embeddings[1])

        cosine_sim = cosine_similarity(embeddings[0], embeddings[1], dim=0).item()
        return cosine_sim

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取{len(data)}条JSON数据")
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return

    new_results = {}

    for sent_id, sent_data in tqdm(data.items()):
        results = {}
        for original_sentence, layer2 in sent_data.items():
            results[original_sentence] = {}

            for original_single_sentence, layer3 in layer2.items():
                results[original_sentence][original_single_sentence] = {}

                valid_candidates = layer3.get("valid_candidates", 0)
                candidate_dict = layer3.get("candidate_simcse", {})

                calculate_simcse1 = {}
                score = calculate_simcse(original_single_sentence,original_single_sentence)
                calculate_simcse1[original_single_sentence] = score
                candidate_dict.update(calculate_simcse1)

                results[original_sentence][original_single_sentence] = {
                    "valid_candidates": valid_candidates,
                    "candidate_simcse": candidate_dict
                }
        new_results[sent_id] = results

    print(f"成功处理{len(new_results)}条数据")

    try:
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(new_results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")

def init_worker(gpu_id, model_name, tokenizer_name, task_queue, result_queue):
    """初始化工作进程，加载模型到指定GPU"""
    try:
        # 设置当前进程使用的GPU
        torch.cuda.set_device(gpu_id)

        # 加载分词器和模型到指定GPU
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name, cache_dir="./cache", local_files_only=True)
        model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir="./cache").to(f"cuda:{gpu_id}")
        model.eval()

        print(f"Worker {gpu_id} 已初始化，使用 GPU: {gpu_id}")

        # 处理任务队列中的任务
        while True:
            task = task_queue.get()
            if task is None:  # 终止信号
                break

            sent_id, sent_data = task
            results = {}

            for original_sentence, clauses in sent_data.items():
                results[original_sentence] = {}

                for clause, candidates in clauses.items():
                    clause_ppl = calculate_ppl(clause, tokenizer, model, gpu_id)

                    candidate_ppl = {}
                    valid_candidates = 0
                    for candidate in candidates:
                        ppl = calculate_ppl(candidate, tokenizer, model, gpu_id)
                        if ppl is not None:
                            candidate_ppl[candidate] = ppl
                            valid_candidates += 1

                    results[original_sentence][clause] = {
                        "clause_ppl": clause_ppl,
                        "candidate_count": len(candidates),
                        "valid_candidates": valid_candidates,
                        "candidate_ppl": candidate_ppl
                    }

            result_queue.put((sent_id, results))

    except Exception as e:
        print(f"Worker {gpu_id} 错误: {str(e)}")
    finally:
        # 清理资源
        del model
        torch.cuda.empty_cache()





def ppl_calculate(input_file, output_file):
    tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
    model = AutoModel.from_pretrained('gpt2', cache_dir="./cache",  local_files_only=True)

    model.to("cuda:0")
    model.eval()

    def calculate_ppl(text):
        input_ids = tokenizer.encode(text, return_tensors='pt').to("cuda:1")
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            ppl = torch.exp(outputs.loss).item()
        return round(ppl, 2)

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取JSON数据，包含 {len(data)} 个原始句子")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {str(e)}")
        return
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return

    new_results = {}
    total_clauses = 0
    processed_clauses = 0
    total_candidates = 0
    processed_candidates = 0

    for sent_id, sent_data in tqdm(data.items()):
        results = {}
        for original_sentence, clauses in sent_data.items():

            results[original_sentence] = {}

            for clause, candidates in clauses.items():
                total_clauses += 1

                clause_ppl = calculate_ppl(clause)

                candidate_ppl = {}
                valid_candidates = 0
                for candidate in candidates:
                    total_candidates += 1
                    ppl = calculate_ppl(candidate)
                    if ppl is not None:
                        candidate_ppl[candidate] = ppl
                        valid_candidates += 1
                        processed_candidates += 1

                results[original_sentence][clause] = {
                    "clause_ppl": clause_ppl,
                    "candidate_count": len(candidates),
                    "valid_candidates": valid_candidates,
                    "candidate_ppl": candidate_ppl
                }
                processed_clauses += 1

        new_results[sent_id] = results

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存结果到JSON文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")

def sort_json_by_key(input_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        sorted_keys = sorted(data.keys(), key=lambda x: int(x))
        sorted_data = {key: data[key] for key in sorted_keys}

        with open(input_file, 'w', encoding='utf-8') as file1:
            json.dump(sorted_data, file1, indent=4)

        print("完成排序")
    except Exception as e:
        print(e)


def filter_ppl_sentence(input_file, output_file, final_sentence_number):
    with open(input_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    new_results = {}
    for sent_id, sent_data in tqdm(data_list.items()):
        results = {}
        for original_sentence, clauses in tqdm(sent_data.items()):

            results[original_sentence] = {}

            for clause, candidates in clauses.items():
                candidate_ppl = candidates.get("candidate_ppl", {})

                # 按值排序并取前k小的项
                sorted_items = sorted(candidate_ppl.items(), key=lambda x: x[1])[:final_sentence_number]
                filtered_ppl = dict(sorted_items)

                results[original_sentence][clause] = {
                    "valid_candidates": 20,
                    "candidate_ppl": filtered_ppl
                }

        new_results[sent_id] = results

    print(f"成功处理{len(new_results)}条数据")

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")


def load_data(dataset=None):
    print(f'__loading__{args.dataset}__')
    train_df = pd.read_csv(f"datasets/{args.dataset}/train.csv", sep=',', on_bad_lines='skip')
    test_df = pd.read_csv(f"datasets/{args.dataset}/dev.csv", sep=',', on_bad_lines='skip')
    train_df.columns = ["index", "sentence", "question", "labels"]
    test_df.columns = ["index", "sentence", "question", "labels"]

    train_df['labels'] = train_df['labels'].map(
        {
            'entailment': 1,
            "not_entailment": 0
        }
    )
    test_df['labels'] = train_df['labels'].map(
        {
            'entailment': 1,
            "not_entailment": 0
        }
    )
    train_df = train_df.dropna(subset=['labels'])
    train_df = train_df.reset_index(drop=True)
    train_df['labels'] = train_df['labels'].astype(int)
    test_df['labels'] = test_df['labels'].astype(int)
    print(train_df['labels'].dtype)
    print(test_df['labels'].dtype)

    return train_df, test_df


class Bert_dataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = BertTokenizer.from_pretrained(f"{args.model_type}", cache_dir='./cache',
                                                       local_files_only=True)

    def __getitem__(self, index):
        #  Pandas DataFrame二维表格 根据行列索引检索
        sentence = self.df.loc[index, 'sentence']

        encoded_dict = self.tokenizer.encode_plus(
            sentence,  # sentence to encode
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_len,
            pad_to_max_length=True,  # 填充到最大长度
            truncation='longest_first',  # 超长时截断策略
            return_attention_mask=True,  # 返回注意力掩码
            return_tensors='pt'  # 返回PyTorch张量
        )

        # 提取编码后的张量
        input_ids = encoded_dict['input_ids'][0]
        attention_mask = encoded_dict['attention_mask'][0]
        token_type_ids = encoded_dict['token_type_ids'][0]
        target = torch.tensor(self.df.loc[index, 'labels'])

        sample = (input_ids, attention_mask, token_type_ids, target)
        return sample

    def __len__(self):
        return len(self.df)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def get_customized_mapping(eps, top_k):
    test_data = load_dataset(path="parquet", data_dir="./datasets/rotten_tomatoes")["validation"]
    test_df = test_data.to_pandas()
    test_df.columns = ["sentence", "labels"]
    test_corpus = " ".join(test_df.sentence)

    corpus = test_corpus

    # 统计语料库中词语频率
    word_freq = [x[0] for x in Counter(corpus.split()).most_common()]

    # 加载词向量
    embedding_path = f"./embeddings/cl100_embeddings.json"
    embeddings = []
    idx2word = []
    word2idx = {}

    print(f"Loading embeddings from {embedding_path}...")

    # 读取JSON格式的词向量文件
    with open(embedding_path, 'r', encoding='utf-8') as file:
        # 假设JSON文件结构为 {word: [vector], word2: [vector2], ...}
        embedding_data = json.load(file)

        # 遍历所有词向量
        for word, vector in embedding_data.items():
            try:
                # 确保向量是浮点类型的列表
                vector = list(map(float, vector))
            except Exception as e:
                print(f"Error loading vector for {word}: {e}")
                continue

            embeddings.append(vector)
            idx2word.append(word)
            word2idx[word] = len(embeddings) - 1

    # 转换为numpy数组
    embeddings = np.array(embeddings, dtype=np.float32)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 转换为PyTorch张量并移至GPU
    embeddings_tensor = torch.tensor(embeddings, device=device, dtype=torch.float32)

    # 标准化词向量（如果不是GloVe）
    if args.embedding_type != "glove.840B.300d":
        norm = torch.norm(embeddings_tensor, dim=1, keepdim=True)
        embeddings_tensor = embeddings_tensor / norm
        print("Embeddings normalized.")

    # 初始化映射表
    word_hash = dict()  # 使用普通字典提高性能
    sim_word_dict = dict()
    p_dict = dict()

    # 为高频词构建映射
    print(f"Building customized mapping for {len(word_freq)} words...")
    for i in trange(len(word_freq)):
        word = word_freq[i]
        if word not in word2idx or word in word_hash:
            continue

        word_idx = word2idx[word]
        word_vector = embeddings_tensor[word_idx].unsqueeze(0)  # [1, dim]

        # 计算最近邻词
        if args.embedding_type == "glove.840B.300d":
            # GPU加速的欧氏距离计算
            dist = torch.cdist(word_vector, embeddings_tensor, p=2)  # 欧氏距离
            index_list = dist.squeeze(0).argsort()[:top_k].cpu().numpy()
        else:
            # GPU加速的余弦相似度计算
            sim = torch.matmul(word_vector, embeddings_tensor.T)  # 余弦相似度
            index_list = sim.squeeze(0).argsort(descending=True)[:top_k].cpu().numpy()

        word_list = [idx2word[x] for x in index_list if x < len(idx2word)]
        embedding_list = embeddings_tensor[index_list].cpu().numpy()  # 移回CPU

        # 处理可能的空列表
        if not word_list:
            continue

        # 根据映射策略处理
        if args.mapping_strategy == "aggressive":
            # 为当前词构建映射
            if args.embedding_type == "glove.840B.300d":
                # 欧氏距离相似度计算
                sim_dist_list = np.sqrt(((word_vector.cpu().numpy() - embedding_list) ** 2).sum(axis=1))
                min_dist = min(sim_dist_list)
                max_dist = max(sim_dist_list)
                if max_dist - min_dist > 1e-8:
                    sim_scores = [-(x - min_dist) / (max_dist - min_dist) for x in sim_dist_list]
                else:
                    sim_scores = [0.0] * len(sim_dist_list)
            else:
                # 余弦相似度归一化
                sim_scores = sim.squeeze(0)[index_list].cpu().numpy()
                min_score = min(sim_scores)
                max_score = max(sim_scores)
                if max_score - min_score > 1e-8:
                    sim_scores = [(x - min_score) / (max_score - min_score) for x in sim_scores]
                else:
                    sim_scores = [0.5] * len(sim_scores)

            # 计算概率分布（GPU加速）
            sim_tensor = torch.tensor(sim_scores, device=device, dtype=torch.float32)

            tmp = torch.exp(eps * sim_tensor / 2).cpu().numpy().astype(np.float64)  # 转高精度NumPy数组
            norm = tmp.sum()
            if norm == 0:
                p_np = np.ones_like(tmp) / len(tmp) if len(tmp) > 0 else np.array([])
            else:
                p_np = tmp / norm  # 高精度归一化
            p = [float(x) for x in p_np]  # 转Python浮点数

            sim_word_dict[word] = word_list

            p_dict[word] = p

    # 保存结果
    os.makedirs(f"./p_dict", exist_ok=True)
    os.makedirs(f"./sim_word_dict", exist_ok=True)

    try:
        with open(f"./p_dict/eps_{args.eps}_top_{args.top_k}.txt",
                  'w') as f:
            json.dump(p_dict, f, ensure_ascii=False, indent=4)
        print(
            f"Probability dictionary saved to p_dict/eps_{args.eps}_top_{args.top_k}.txt")
    except Exception as e:
        print(f"Failed to save p_dict: {e}")

    try:
        with open(f"./sim_word_dict/eps_{args.eps}_top_{args.top_k}.txt",
                  'w') as f:
            json.dump(sim_word_dict, f, ensure_ascii=False, indent=4)
        print(
            f"Similar word dictionary saved to sim_word_dict=/eps_{args.eps}_top_{args.top_k}.txt")
    except Exception as e:
        print(f"Failed to save sim_word_dict: {e}")

    return sim_word_dict, p_dict


def load_local_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f if line.strip()])


def generate_new_sentences(df, sim_word_dict, p_dict, question_path, sentence_path, sentence_number):
    SENTENCE_SPLIT_REGEX = re.compile(r'([^?!。]+[?!。]?)')

    def split_into_subsentences(text):
        """将文本按句号、问号、感叹号分割成子句"""
        text = text.strip()
        if not text:
            return []
        subsentences = [s.strip() for s in SENTENCE_SPLIT_REGEX.findall(text) if s.strip()]
        return subsentences

    def generate_candidates(text, s_number):
        """为单个文本生成200个候选版本"""
        punct = list(string.punctuation)
        stop_words = load_local_stopwords("./nltk_data/corpora/stopwords/english")
        candidates = []
        original_words = text.split()

        for _ in range(s_number):
            new_words = []
            for word in original_words:
                if (word in stop_words) or (word in punct) or (word not in sim_word_dict):
                    if is_number(word):
                        try:
                            word = str(round(float(word)) + np.random.randint(1000))
                        except:
                            pass
                    new_words.append(word)
                else:
                    p = p_dict[word]
                    # 实现随机选择
                    new_word = np.random.choice(sim_word_dict[word])
                    new_words.append(new_word)

            candidates.append(" ".join(new_words))
        return candidates

    # 使用ID作为键，保留所有重复记录
    sentence_candidates = {}
    for i in trange(len(df.sentence), desc="生成sentence候选"):
        original_dict = {}
        original_text = df.sentence[i]
        sentence_id = df.index[i]
        subsentences = split_into_subsentences(original_text)

        subsentence_candidates = {}
        for subsent in subsentences:
            subsentence_candidates[subsent] = generate_candidates(subsent, sentence_number)

        # 使用ID作为键，而不是原始文本
        original_dict[original_text] = subsentence_candidates
        sentence_candidates[sentence_id] = original_dict

    if not os.path.exists(os.path.dirname(sentence_path)):
        os.makedirs(os.path.dirname(sentence_path), exist_ok=True)

    # 保存为JSON文件
    with open(sentence_path, "w", encoding="utf-8") as f:
        json.dump(sentence_candidates, f, ensure_ascii=False, indent=2)

    print(f"已保存 {len(sentence_candidates)} 个句子")
    return df


def generate_new_sents_s1(df, sim_word_dict, p_dict, save_stop_words):
    punct = list(string.punctuation)
    # nltk.download('punkt')
    stop_words = load_local_stopwords("./nltk_data/corpora/stopwords/english")

    sentence = df.sentence
    new_sentence = []

    for i in trange(len(sentence)):
        record = sentence[i].split()
        new_record = []
        for word in record:
            # 判断word是否在冠词，标点符号里，或者不在词表里
            if (word in stop_words) or (word in punct) or (word not in sim_word_dict):
                if is_number(word):
                    try:
                        # 数字进行随机扰动
                        word = str(round(float(word)) + np.random.randint(1000))
                    except:
                        pass
                new_record.append(word)
            else:
                new_word = np.random.choice(sim_word_dict[word])[0]
                new_record.append(new_word)
        new_sentence.append(" ".join(new_record))

    df.sentence = new_sentence

    question = df.question
    new_question = []

    for i in trange(len(question)):
        record = question[i].split()
        new_record = []
        for word in record:
            if (save_stop_words and word in stop_words) or (word not in sim_word_dict):
                if is_number(word):
                    try:
                        word = str(round(float(word)) + np.random.randint(1000))
                    except:
                        pass
                new_record.append(word)
            else:
                p = p_dict[word]
                new_word = np.random.choice(sim_word_dict[word], 1, p=p)[0]
                new_record.append(new_word)
        new_question.append(" ".join(new_record))

    df.question = new_question

    if not os.path.exists(
            f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}"):
        os.makedirs(
            f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}")
    if type == "train":
        df.to_csv(
            f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}/train.tsv",
            "\t", index=0)
    else:
        df.to_csv(
            f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}/test.tsv",
            "\t", index=0)

    return df
