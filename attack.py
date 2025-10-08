import os
import random
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import torch
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
import torch
from transformers import BertTokenizer, BertForMaskedLM
from rouge import Rouge

parser = get_parser()
args = parser.parse_args()


class BertMaskAttack:
    def __init__(self, model_name='bert-base-uncased', max_length=512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name, unk_token='[UNK]', cache_dir="./cache",
                                                       local_files_only=True)
        self.model = BertForMaskedLM.from_pretrained(model_name, cache_dir="./cache", local_files_only=True).to(
            self.device)
        self.model.eval()
        self.rouge = Rouge()
        self.mask_token = self.tokenizer.mask_token
        self.mask_id = self.tokenizer.mask_token_id
        self.max_length = max_length

    def predict_masked_token(self, text, mask_position):
        """预测指定位置被掩码后的token"""
        tokens = self.tokenizer.tokenize(text)

        # 检查token长度和mask位置有效性
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        if mask_position >= len(tokens):
            return None, None

        original_token = tokens[mask_position]
        tokens[mask_position] = self.mask_token

        # 转换为模型输入格式
        inputs = self.tokenizer.convert_tokens_to_ids(tokens)
        inputs = torch.tensor([inputs]).to(self.device)

        # 模型推理
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = outputs.logits[0, mask_position]

        # 获取预测概率最高的token
        predicted_index = torch.argmax(predictions).item()
        predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[0]

        return predicted_token, original_token

    def run_attack(self, original_text, perturbed_text):
        """执行掩码攻击并计算token级别的成功率"""
        # 分词原始文本和扰动文本
        original_tokens = self.tokenizer.tokenize(original_text)
        perturbed_tokens = self.tokenizer.tokenize(perturbed_text)

        # 截断超长文本
        if len(original_tokens) > self.max_length:
            original_tokens = original_tokens[:self.max_length]
        if len(perturbed_tokens) > self.max_length:
            perturbed_tokens = perturbed_tokens[:self.max_length]

        # 确保两个分词序列长度一致（必要时截断）
        min_len = min(len(original_tokens), len(perturbed_tokens))
        original_tokens = original_tokens[:min_len]
        perturbed_tokens = perturbed_tokens[:min_len]

        # 初始化预测token列表
        predicted_tokens = perturbed_tokens.copy()

        # 记录成功恢复的token数量和总token数量
        total_tokens = 0
        success_tokens = 0

        # 对扰动文本的每个token进行掩码预测
        for i in range(len(perturbed_tokens)):
            # 创建掩码文本
            masked_tokens = predicted_tokens.copy()
            masked_tokens[i] = self.mask_token
            masked_text = self.tokenizer.convert_tokens_to_string(masked_tokens)

            # 预测掩码位置的token
            predicted_token, _ = self.predict_masked_token(masked_text, i)

            if predicted_token is not None:
                # 更新预测token列表
                predicted_tokens[i] = predicted_token

                # 计算攻击成功率（与原始token比较）
                total_tokens += 1
                if predicted_token == original_tokens[i]:
                    success_tokens += 1

        # 计算token级别的攻击成功率
        attack_success_rate = success_tokens / total_tokens if total_tokens > 0 else 0.0

        # 转换回文本格式
        predicted_text = self.tokenizer.convert_tokens_to_string(predicted_tokens)

        return predicted_text, original_text, attack_success_rate

    def evaluate_attack(self, original_text, predicted_text):
        """使用Rouge-L F1分数评估攻击效果"""
        if predicted_text is None:
            return 0.0
        try:
            scores = self.rouge.get_scores(predicted_text, original_text)
        except Exception as e:
            return None

        return scores[0]['rouge-l']['f']


def bert_embedding_attack(origin_df, sub_df):
    attacker = BertMaskAttack()

    attack_success_list = []
    avg_rouge_f1 = 0.0
    total = len(origin_df)
    for i in trange(len(sub_df.sentence), desc="针对sentence进行攻击"):
        perturbed_text = sub_df.sentence[i]
        origin_text = origin_df.sentence[i]

        # 攻击预测bert生成的文本
        predicted_text, original_text, attack_rate = attacker.run_attack(origin_text, perturbed_text)
        attack_success_list.append(attack_rate)

        rouge_l_f1 = attacker.evaluate_attack(original_text, predicted_text)
        if rouge_l_f1 is None:
            total -= 1
        else:
            avg_rouge_f1 += rouge_l_f1

    avg_rouge_f1 = avg_rouge_f1 / total
    average_success_rate = np.array(attack_success_list).mean()
    print(f"攻击平均成功率为：{average_success_rate}")
    print(f"攻击的rouge_f1分数为：{avg_rouge_f1}")

    return avg_rouge_f1, average_success_rate


def load_local_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f if line.strip()])


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


def knn_attack(origin_df, sub_df):
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

    # 设备配置（优化：全局GPU张量）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings_tensor = torch.tensor(embeddings, device=device, dtype=torch.float32)

    # 加载停用词和标点（全局加载）
    punct = set(string.punctuation)
    stop_words = load_local_stopwords("./nltk_data/corpora/stopwords/english")

    def calculate_attack_accuracy_batch(sub_texts, origin_texts):
        token1 = 0
        token2 = 0

        for sub_text, origin_text in zip(sub_texts, origin_texts):
            sub_list = sub_text.split()
            origin_list = origin_text.split()

            # 收集需要处理的单词（过滤停用词、标点和数字）
            valid_pairs = []
            for sub_word, origin_word in zip(sub_list, origin_list):
                if sub_word in punct or sub_word in stop_words or is_number(sub_word):
                    continue
                if sub_word in word2idx:  # 确保词在词表中
                    valid_pairs.append((sub_word, origin_word))

            if not valid_pairs:
                continue

            # 批量获取词向量（优化：减少GPU调用）
            sub_words, origin_words = zip(*valid_pairs)
            word_indices = [word2idx[w] for w in sub_words]
            query_vectors = embeddings_tensor[word_indices]  # 形状：[batch_size, embedding_dim]

            # 批处理KNN（优化：一次性计算所有距离）
            dist = torch.cdist(query_vectors, embeddings_tensor, p=2)  # [batch_size, vocab_size]
            top_indices = dist.argsort(dim=1)[:, :args.attack_top_k].cpu().numpy()  # [batch_size, top_k]

            # 统计成功攻击的token
            for i, (sub_word, origin_word) in enumerate(valid_pairs):
                neighbors = [idx2word[idx] for idx in top_indices[i] if idx < len(idx2word)]
                if origin_word in neighbors:
                    token2 += 1
                token1 += 1

        return token1, token2

    # 分批次处理数据（优化：使用更大的批处理）
    batch_size = 64
    all_token, attack_token = 0, 0

    # 处理sentence列
    for i in trange(0, len(sub_df.sentence), batch_size, desc="针对sentence进行攻击"):
        batch_sub = sub_df.sentence[i:i + batch_size]
        batch_origin = origin_df.sentence[i:i + batch_size]
        count1, count2 = calculate_attack_accuracy_batch(batch_sub, batch_origin)
        all_token += count1
        attack_token += count2

    print(f"攻击总的token数量为：{all_token}")
    print(f"攻击成功的token数量为：{attack_token}")
    print(f"攻击成功率为：{attack_token/all_token}")
    return all_token, attack_token

# def knn_attack(origin_df, sub_df):
#     # 获取词表
#     embedding_path = f"./embeddings/{args.embedding_type}.txt"
#     embeddings = []
#     idx2word = []
#     word2idx = {}
#
#     print(f"Loading embeddings from {embedding_path}...")
#     with open(embedding_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             parts = line.strip().split(" ")
#             if len(parts) < 301:  # 至少需要1个单词+300个数值
#                 continue
#             word = parts[0]
#             try:
#                 vector = list(map(float, parts[1:]))
#             except Exception as e:
#                 print(f"Error loading vector for {word}: {e}")
#                 continue
#
#             embeddings.append(vector)
#             idx2word.append(word)
#             word2idx[word] = len(embeddings) - 1
#
#     embeddings = np.array(embeddings, dtype=np.float32)
#     # 设备配置
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     punct = list(string.punctuation)
#     stop_words = load_local_stopwords("./nltk_data/corpora/stopwords/english")
#     # 转换为PyTorch张量并移至GPU
#     embeddings_tensor = torch.tensor(embeddings, device=device, dtype=torch.float32)
#
#     def calculate_attack_accuracy(sub_text, origin_text):
#         total_token = 0
#         victory_token = 0
#
#         sub_list = sub_text.split()
#         origin_list = origin_text.split()
#
#         for sub_word, origin_word in zip(sub_list, origin_list):
#             if sub_word in punct or sub_word in stop_words or is_number(sub_word):
#                 continue
#             else:
#                 word_idx = word2idx[word]
#                 word_vector = embeddings_tensor[word_idx].unsqueeze(0)
#
#                 # 欧氏距离
#                 dist = torch.cdist(word_vector, embeddings_tensor, p=2)
#                 index_list = dist.squeeze(0).argsort()[:args.attack_top_k].cpu().numpy()
#                 word_list = [idx2word[x] for x in index_list if x < len(idx2word)]
#                 if origin_word in word_list:
#                     victory_token += victory_token
#                 total_token += total_token
#
#         return total_token, victory_token
#
#     all_token = 0
#     attack_token = 0
#     for i in trange(len(sub_df.sentence), desc="针对sentence进行攻击"):
#         sub_text = sub_df.sentence[i]
#         origin_text = origin_df.sentence[i]
#         count1, count2 = calculate_attack_accuracy(sub_text, origin_text)
#         all_token += count1
#         attack_token += count2
#
#     for i in trange(len(sub_df.question), desc="针对question进行攻击"):
#         sub_text = sub_df.question[i]
#         origin_text = origin_df.question[i]
#         count1, count2 = calculate_attack_accuracy(sub_text, origin_text)
#         all_token += count1
#         attack_token += count2
#
#     print(f"攻击总的token数量为：{all_token}\n")
#     print(f"攻击成功的token数量为：{attack_token}")
#     return all_token, attack_token
