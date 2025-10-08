import logging
import time
import tqdm
from openai import OpenAI
import requests
import os
import pandas as pd
from datasets import load_dataset

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


class APIClient:
    def __init__(self, api_key, url, model_name):
        self.api_key = api_key
        self.url = url
        self.model_name = model_name
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        self.request_count = 0;
        self.max_request_per_minute = 10

    def _rate_limit(self):
        self.request_count += 1
        if self.request_count % self.max_request_per_minute == 0:
            time.sleep(60)
        else:
            time.sleep(3)

    def generate_response(self, prompt, temperature=0.7, retry=3):
        for attempt in range(retry):
            try:
                self._rate_limit()
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 3000
                }

                res = requests.post(f"{self.url}/chat/completions", headers=self.headers, json=payload, timeout=120)

                if res.status_code == 200:
                    response_data = res.json()
                    # 确保响应格式正确
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        content = response_data['choices'][0]['message']['content']
                        if content.strip() in ['0', '1']:
                            return content
                        else:
                            logger.warning(f"API返回非预期内容: '{content}'")
                    else:
                        logger.warning(f"API响应格式异常: {response_data}")
            except requests.exceptions.ReadTimeout:
                logger.warning("请求超时，重试...")
            except Exception as e:
                logger.error(f"API调用异常：{e}")
                break
        return None


def make_qnli_prompt_binary(sentence: str) -> str:
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


def judgment_gpt(sentence):
    # genrate_client = APIClient(
    #     api_key="",
    #     url="https://api.openai.com/v1", model_name="gpt-4o-2024-11-20")
    genrate_client = APIClient(
        api_key="",
        url="https://dashscope.aliyuncs.com/compatible-mode/v1", model_name="qwen3-max-preview")



    prompt = make_qnli_prompt_binary(sentence)

    response = genrate_client.generate_response(prompt)
    response_clean = response.strip()

    if response_clean not in ["0", "1"]:
        return None  # 非法结果
    return int(response_clean)


def calculate_classify_accuracy(df):
    total = 0
    right_count = 0
    error_count = 0
    for i in tqdm.trange(len(df.sentence), desc="gpt分类准确率判断"):
        try:
            predicted = judgment_gpt(df.sentence[i])
            if predicted is None:
                continue  # 不计入 total
            total += 1
            if predicted == df.labels[i]:
                right_count += 1
            else:
                error_count += 1
        except Exception as e:
            logger.error(f"第 {i} 条样本判断失败：{e}")
            continue
    print(f"gpt4-o预测正确率为：{(right_count / total):.4f}")
    print(f"gpt4-o预测错误率为：{(error_count / total):.4f}")
    logger.info(f"最终分类预测正确率为：{(right_count / total): .4f},最终分类预测错误率为：{(error_count / total):.4f}")
    return None


if __name__ == "__main__":
    data = load_dataset(path="parquet", data_dir="./datasets")
    test_data = data["validation"]

    test_data = pd.DataFrame(test_data)
    test_data.columns = ["question", "sentence", "labels", "index"]
    calculate_classify_accuracy(test_data)
