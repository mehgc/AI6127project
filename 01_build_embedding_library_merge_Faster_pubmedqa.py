import os
import json
import re
import pandas as pd
from datasets import Dataset, load_dataset
import torch
import time
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer  # ✅ 保留这个用于 tokenizer 截断
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from more_itertools import chunked


# ====== ✅ 统一路径配置 ======
SAVE_DIR = "/home/msai/ghu003/6127project_experiments/outputs"  # 所有保存的文件都放在这个目录下
os.makedirs(SAVE_DIR, exist_ok=True)

EMBEDDING_OUTPUT_PATH = os.path.join(SAVE_DIR, "all_datasets_strict_embeddings.json")






# ====== ✅ 实用工具函数 ======

def is_case_based(text):
    """判断是否为病例类问题"""
    case_keywords = ["year-old", "patient", "presents with", "admitted", "history", "examination", "symptoms", "complains of"]
    return any(kw in text.lower() for kw in case_keywords)

def average_option_length(options: list) -> float:
    """计算选项的平均长度"""
    return sum(len(opt['value']) for opt in options) / len(options) if len(options) == 4 else 0

# ====== ✅ 对于pubmd_qa，生成Few-shot Prompt（Prompt + Cot ） ======
def format_pubmedqa_with_longanswer_cot(example):
    """
    使用 long_answer 构造 PubMedQA 的带 CoT 的 Prompt。
    自动处理字段缺失或异常，确保格式整洁。
    """
    # 安全提取字段
    context_list = example.get("context", {}).get("contexts", [])
    question = example.get("question", "").strip()
    final_decision = example.get("final_decision", "Maybe").strip().capitalize()
    cot = example.get("long_answer", "").strip()

    # 清理上下文并拼接
    context = "\n".join([c.strip() for c in context_list if c.strip()])

    # 格式化 Prompt
    prompt = f"""### Question:
{question}

### Context:
{context}

### Let's think step by step:
{cot}

### Answer:
{final_decision}"""

    return {"input": prompt, "output": final_decision}


# === ✅ 拆分加载函数：分别加载三个净果筛选的数据集
def load_filtered_pubmedqa(cache_path="/home/msai/ghu003/6127project_experiments/outputs/pubmedqa_with_cot.json", max_samples=8000):
    if os.path.exists(cache_path):
        print(f"📄 已检测到缓存文件：{cache_path}，直接加载")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    dataset = load_dataset("pubmed_qa", "pqa_artificial", split="train")

    # ✅ 第一步：筛选出符合条件的数据
    filtered = [
        format_pubmedqa_with_longanswer_cot(ex) for ex in dataset
        if ex['final_decision'] in ['yes', 'no', 'maybe'] and 'long_answer' in ex
    ]

    # ✅ 第二步：若超过 max_samples 条，则随机抽样
    if len(filtered) > max_samples:
        print(f"🔎 共筛选出 {len(filtered)} 条，超过 {max_samples}，将随机抽取")
        filtered = random.sample(filtered, max_samples)

    # ✅ 第三步：保存到缓存文件
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
        print(f"✅ 保存缓存文件：{cache_path}")

    return filtered

# ====== ✅ 嵌入向量库构建并保存 ======

def extract_cot_from_input(prompt: str):
    """
    从完整的 prompt 中提取出 CoT（Chain-of-Thought）部分的内容。

    参数：
    - prompt (str): 格式化后的完整 Prompt，包含 "### Let's think step by step:" 和 "### Answer:" 标签。

    返回：
    - cot (str): 被提取出来的思维链（CoT）文本，如果无法匹配则返回空字符串。
    """
    match = re.search(
        r"### Let's think step by step:\n(.+?)\n+### Answer:",
        prompt,
        re.DOTALL  # 允许换行符参与匹配
    )
    return match.group(1).strip() if match else ""

# ==⚠️⚠️⚠️⚠️⚠️⚠️⚠️【需要调整】⚠️⚠️⚠️⚠️⚠️⚠️⚠️ =====
def build_embedding_library(data, output_path, model_name="all-MiniLM-L6-v2", max_tokens=512):
    """
    构建嵌入向量库（支持 GPU + 分批编码 + 日志输出）

    参数：
        - data: 包含 input/output 的样本数据（已含 prompt + cot）
        - output_path: 保存 JSON 的路径
        - model_name: 嵌入模型名
        - max_tokens: 截断 token 长度
    """
    print(f"🧠 开始构建嵌入库，总共样本数：{len(data)} 条")

    # ✅ 1. 自动选择设备和 batch size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ==⚠️⚠️⚠️⚠️⚠️⚠️⚠️【需要调整】⚠️⚠️⚠️⚠️⚠️⚠️⚠️ =====
    # 如果你使用的是 6GB 显存（例如 RTX 2060），建议把 batch_size = 64 改成 batch_size = 16 或 32；
    # batch_size = 128  # 或 256，如果是 A100 等高显存 GPU
    batch_size = 64 if device == 'cuda' else 16
    print(f"🚀 使用设备：{device.upper()}，batch_size={batch_size}")

    # ✅ 2. 初始化模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SentenceTransformer(model_name, device=device)

    # ✅ 3. 提取 prompt 前缀并进行 token 截断
    inputs = []
    for item in data:
        full_prompt = item['input']
        head = full_prompt.split("### Let's think step by step:")[0].strip() \
            if "### Let's think step by step:" in full_prompt else full_prompt
        tokens = tokenizer(head, truncation=True, max_length=max_tokens)
        token_ids = tokens['input_ids']
        token_strings = tokenizer.convert_ids_to_tokens(token_ids)
        truncated_text = tokenizer.convert_tokens_to_string(token_strings)

        inputs.append(truncated_text)

    # ✅ 4. 手动分批 encode（带日志）
    print("📦 正在分批进行嵌入编码...")
    start_total = time.time()
    all_embeddings = []

    text_chunks = list(chunked(inputs, batch_size))
    for i, chunk in enumerate(tqdm(text_chunks, desc="🔁 编码中")):
        start_batch = time.time()
        emb = model.encode(chunk)
        all_embeddings.append(emb)
        print(f"✅ Batch {i+1}/{len(text_chunks)} done in {time.time() - start_batch:.2f}s")

    # ✅ 5. 合并所有嵌入
    embeddings = np.vstack(all_embeddings)
    print(f"✅ 全部编码完成，用时：{time.time() - start_total:.2f} 秒")

    # ✅ 6. 构建保存结构
    result = []
    for i, item in enumerate(data):
        result.append({
            "id": f"Q{i}",
            "input_short": inputs[i],
            "output": item["output"],
            "cot": extract_cot_from_input(item["input"]),
            "embedding": embeddings[i].tolist()
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"💾 嵌入库保存成功：{output_path}")

    return result, inputs

# ====== ✅ 主程序入口 ======

if __name__ == "__main__":
    print("🚀 加载筛选后的数据集...")

    pubmedqa_data = load_filtered_pubmedqa()
    print(f"✅ PubMedQA 条数：{len(pubmedqa_data)}")

    output_path = os.path.join(SAVE_DIR, "emb_pubmedqa.json")
    print(f"\n=== 🧩 构建嵌入库：PubMedQA（共 {len(pubmedqa_data)} 条） ===")
    build_embedding_library(pubmedqa_data, output_path)