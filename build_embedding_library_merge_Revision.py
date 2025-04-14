import os
import json
import re
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ====== ✅ 统一路径配置 ======
SAVE_DIR = "/home/msai/ghu003/6127project_experiments/outputs/"  # 所有保存的文件都放在这个目录下
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

# ====== ✅ 各数据集格式转换 ======

def format_medqa(example):
    question = example['question']
    options_str = "\n".join([f"({opt['key']}, '{opt['value']}')" for opt in example['options']])
    prompt = f"### Question:\n{question}\n\n### Options:\n{options_str}"
    return {"input": prompt, "output": example['answer_idx']}

def format_medmcqa(example):
    question = example['question']
    options = [example['opa'], example['opb'], example['opc'], example['opd']]
    options_str = "\n".join([f"({chr(65+i)}, '{opt}')" for i, opt in enumerate(options)])
    prompt = f"### Question:\n{question}\n\n### Options:\n{options_str}"
    return {"input": prompt, "output": chr(65 + example['cop'])}

def format_pubmedqa(example):
    prompt = f"### Question:\n{example['question']}\n\n### Context:\n{example['context']}\n\n### Answer:\nYes/No/Maybe"
    return {"input": prompt, "output": example['final_decision'].capitalize()}


# ====== ✅ 加载、筛选并格式化所有训练集样本 ======

def load_and_filter_datasets(min_q_len=500, min_opt_len=30, verbose=True, max_per_dataset=2000):
    """
    加载并筛选 MedQA / MedMCQA / PubMedQA 三个数据集
    每个数据集最多保留 max_per_dataset 条（超过就随机抽样）
    返回筛选后统一格式的数据 + 每个数据集的统计量
    """
    print("📥 正在加载数据集...")
    medqa_raw = load_dataset("bigbio/med_qa", "med_qa_en_4options_source", trust_remote_code=True)['train']
    medmcqa_raw = load_dataset("medmcqa")['train']
    pubmedqa_raw = load_dataset("pubmed_qa", "pqa_artificial", split="train")

    all_data = []
    stats = {"medqa": 0, "medmcqa": 0, "pubmedqa": 0}

    # ====== MedQA ======
    medqa_data = [
        format_medqa(ex) for ex in medqa_raw
        if len(ex['question']) > min_q_len and
           average_option_length(ex['options']) > min_opt_len and
           is_case_based(ex['question'])
    ]
    if len(medqa_data) > max_per_dataset:
        medqa_data = random.sample(medqa_data, max_per_dataset)  # ✅ 超过8000则抽样
    stats["medqa"] = len(medqa_data)
    all_data.extend(medqa_data)

    # ====== MedMCQA ======
    medmcqa_data = [
        format_medmcqa(ex) for ex in medmcqa_raw
        if len(ex['question']) > min_q_len and
           average_option_length([
               {"key": "A", "value": ex['opa']}, {"key": "B", "value": ex['opb']},
               {"key": "C", "value": ex['opc']}, {"key": "D", "value": ex['opd']}
           ]) > min_opt_len and
           is_case_based(ex['question'])
    ]
    if len(medmcqa_data) > max_per_dataset:
        medmcqa_data = random.sample(medmcqa_data, max_per_dataset)  # ✅ 超过8000则抽样
    stats["medmcqa"] = len(medmcqa_data)
    all_data.extend(medmcqa_data)

    # ====== PubMedQA ======
    pubmedqa_data = [
        format_pubmedqa(ex) for ex in pubmedqa_raw
        if ex['final_decision'] in ['yes', 'no', 'maybe']
    ]
    if len(pubmedqa_data) > max_per_dataset:
        pubmedqa_data = random.sample(pubmedqa_data, max_per_dataset)  # ✅ 超过8000则抽样
    stats["pubmedqa"] = len(pubmedqa_data)
    all_data.extend(pubmedqa_data)

    if verbose:
        print(f"✅ 筛选完成：MedQA: {stats['medqa']} 条, MedMCQA: {stats['medmcqa']} 条, PubMedQA: {stats['pubmedqa']} 条")
        print(f"🎯 总计 {len(all_data)} 条高质量样本")

    return all_data, stats

# ====== ✅ 嵌入向量库构建并保存 ======

def build_embedding_library(data, output_path, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    print(f"🧠 正在对 {len(data)} 条样本进行编码...")
    embeddings = model.encode([item['input'] for item in data], show_progress_bar=True)

    result = []
    for i, item in enumerate(data):
        result.append({
            "id": f"Q{i}",
            "input": item["input"],
            "output": item["output"],
            "embedding": embeddings[i].tolist()
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"💾 嵌入库保存至：{output_path}")

# ====== ✅ 主程序入口 ======

if __name__ == "__main__":
    all_data, stats = load_and_filter_datasets()
    build_embedding_library(all_data, EMBEDDING_OUTPUT_PATH)
