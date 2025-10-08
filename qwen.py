import logging
import time
import tqdm

from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import os
import pandas as pd
from datasets import load_dataset
from rouge import Rouge

os.makedirs("log2", exist_ok=True)

log_filename = os.path.join("log2", time.strftime(f"request_QNLI.log"))

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def make_qnli_prompt_binary(sentence: str) -> str:
    # You are a highly skilled sentiment analysis expert. 
    # Your task is to classify the sentiment of sentence as either positive or negative.

    # You are a helpful assistant and an expert in sentiment analysis.
    # Classify if the sentence is positive or negative sentiment.
    # You are a sentiment analysis expert specialized in movie reviews. 
    # Your task is to classify the sentiment of a given sentence from the Stanford Sentiment Treebank (SST-2) dataset. 
    # The dataset contains binary sentiment labels (positive/negative). 

    # Respond with:
    # 0 if the sentence is negative.
    # 1 if the sentence is positive.
    
    # Sentence: {sentence}
    # Answer (only 0 or 1)
    return f"""
    You are a highly skilled sentiment analysis expert. 
    Your task is to classify the sentiment of moive review as either positive or negative.

    Respond with:
    0 if the sentence is negative.
    1 if the sentence is positive.

    Examples: 

    Review: it's a charming and often affecting journey .\n
    Sentiment: 1

    Review: unflinchingly bleak and desperate.\n
    Sentiment: 0

    now, answer the following:
    Review: {sentence}
"""


def evaluate_attack_F1(original_text, predicted_text):
    """使用Rouge-L F1分数评估攻击效果"""
    if predicted_text is None:
        return 0.0
    try:
        rouge = Rouge()
        scores = rouge.get_scores(hyps=predicted_text, refs=original_text)
    except Exception as e:
        return None

    return scores[0]['rouge-l']['f']

def extract_label(response: str):
    # 简单提取逻辑：检查是否包含0或1
    if "0" in response and "1" not in response:
        return 0
    elif "1" in response and "0" not in response:
        return 1
    # 更复杂的逻辑可以使用正则表达式或关键词匹配
    elif "negative" in response.lower() or "no" in response.lower():
        return 0
    elif "positive" in response.lower() or "yes" in response.lower():
        return 1
    return None


def calculate_classify_accuracy(df, eps):
    device = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B-Instruct",
        cache_dir='./cache',
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct", cache_dir="./cache")

    total = len(df.sentence)
    right_count = 0
    error_count = 0
    for i in tqdm.trange(len(df.sentence), desc="qwen分类准确率判断"):
        prompt = make_qnli_prompt_binary(df.sentence[i])
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        
        answer = extract_label(response)
        if answer is None:
            # print(f"\nsentence:{df.sentence[i]}")
            total -= 1
            continue
        # print(type(int(df.labels[i])), type(answer), int(df.labels[i]) == answer)
        if int(df.labels[i]) == answer:
            right_count += 1
        else:
            error_count += 1

    logger.info(f"隐私预算为{eps}的最终分类预测正确率为：{(right_count / total): .4f},最终分类预测错误率为：{(error_count / total):.4f}")
    print(f"qwen2-1.5B预测正确率为：{(right_count / total):.4f}")
    print(f"qwen2-1.5B预测错误率为：{(error_count / total):.4f}")


def calculate_F1(test_data, origin_test_data, eps):
    total = len(test_data.sentence)
    avg_rouge_f1 = 0.0
    for i in tqdm.trange(len(test_data.sentence), desc="计算F1相似度得分"):
        perturbed_text = test_data.sentence[i]
        origin_text = origin_test_data.sentence[i]

        rouge_l_f1 = evaluate_attack_F1(origin_text, perturbed_text)
        if rouge_l_f1 is None:
            total -= 1
        else:
            avg_rouge_f1 += rouge_l_f1

    avg_rouge_f1 = avg_rouge_f1 / total
    logger.info(f"隐私预算为{eps}的最终F1得分为：{(avg_rouge_f1):.4f}")
    logger.info(f"----------------------------------------")
