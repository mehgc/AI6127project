import os
import json
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from tqdm import tqdm

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ✅ 找出最相似的 top-k 示例
def find_top_k_similar_examples(query, embedding_file, top_k=3, model_name="all-MiniLM-L6-v2"):
    with open(embedding_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = SentenceTransformer(model_name)
    query_embedding = model.encode(query)
    all_embeddings = [item["embedding"] for item in data]
    scores = util.cos_sim(query_embedding, all_embeddings)[0]
    top_results = scores.topk(k=top_k)

    similar_examples = []
    for score, idx in zip(top_results.values, top_results.indices):
        matched = data[idx]
        similar_examples.append({
            "input": matched["input"],
            "output": matched["output"]
        })
    return similar_examples

# ✅ 构造 Few-shot Prompt
def construct_prompt(examples, query):
    prompt = ""
    for ex in examples:
        prompt += f"{ex['input']}\n\n### Answer:\n{ex['output']}\n\n---\n\n"
    prompt += query + "\n\n### Answer:"
    return prompt

def get_query_prompt_without_answer(example, dataset_type):
    if dataset_type == "medqa":
        question = example['question']
        options_str = "\n".join([f"{opt['key']}: {opt['value']}" for opt in example['options']])
    elif dataset_type == "medmcqa":
        if example['cop'] not in [0, 1, 2, 3]:
            return None
        question = example['question']
        options_str = f"A: {example['opa']}\nB: {example['opb']}\nC: {example['opc']}\nD: {example['opd']}"
    elif dataset_type == "pubmedqa":
        question = example['question']
        options_str = "A: Yes\nB: No\nC: Maybe"
    else:
        raise ValueError(f"❌ 不支持的数据集类型: {dataset_type}")

    prompt = f"""Below is a medical question with multiple choice options. Provide only the letter of the correct answer (e.g., A, B, C, D).

### Question:
{question}

### Options:
{options_str}"""
    return prompt


# ✅ 数据集格式化函数们
def format_medqa(example):
    question = example['question']
    options_str = "\n".join([f"{opt['key']}: {opt['value']}" for opt in example['options']])
    correct_answer_idx = example['answer_idx']
    prompt = f"""Below is a medical question with multiple choice options. Provide only the letter of the correct answer (e.g., A, B, C, D).

### Question:
{question}

### Options:
{options_str}

### Answer:
{correct_answer_idx}"""
    return {"text": prompt, "answer_idx": correct_answer_idx}

def format_medmcqa(example):
    cop_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    if example['cop'] not in [0, 1, 2, 3]:
        return None
    question = example['question']
    options_str = f"A: {example['opa']}\nB: {example['opb']}\nC: {example['opc']}\nD: {example['opd']}"
    correct_answer_idx = cop_to_letter[example['cop']]
    prompt = f"""Below is a medical question with multiple choice options. Provide only the letter of the correct answer (e.g., A, B, C, D).

### Question:
{question}

### Options:
{options_str}

### Answer:
{correct_answer_idx}"""
    return {"text": prompt, "answer_idx": correct_answer_idx}

def format_pubmedqa(example):
    question = example['question']
    options_str = "A: Yes\nB: No\nC: Maybe"
    answer_map = {"yes": "A", "no": "B", "maybe": "C"}
    correct_answer_idx = answer_map[example['final_decision'].lower()]
    prompt = f"""Below is a medical question with multiple choice options. Provide only the letter of the correct answer (e.g., A, B, C, D).

### Question:
{question}

### Options:
{options_str}

### Answer:
{correct_answer_idx}"""
    return {"text": prompt, "answer_idx": correct_answer_idx}

def format_inference_prompt(example, dataset_type):
    if dataset_type == "medqa":
        return format_medqa(example)["text"]
    elif dataset_type == "medmcqa":
        formatted = format_medmcqa(example)
        return formatted["text"] if formatted else None
    elif dataset_type == "pubmedqa":
        return format_pubmedqa(example)["text"]
    else:
        raise ValueError(f"❌ 不支持的数据集类型: {dataset_type}")

# ✅ 可视化调试函数
def show_first_prompt_and_examples(dataset, dataset_type, embedding_file):
    print("\n====================== 🔍 Prompt Debug Info ======================")
    example = dataset[0]
    query = get_query_prompt_without_answer(example, dataset_type)
    print("📌 Query:\n", query)

    top_examples = find_top_k_similar_examples(query, embedding_file)
    print("\n📚 Top-3 Retrieved Examples:")
    for i, ex in enumerate(top_examples, 1):
        print(f"\nExample #{i}")
        print(f"Input:\n{ex['input']}\n\nOutput:\n{ex['output']}\n{'-'*50}")

    full_prompt = construct_prompt(top_examples, query)
    print("\n🧾 Full Prompt to Model:\n")
    print(full_prompt)
    print("\n=================================================================\n")

def main():
    # 参数配置
    embedding_file = Path(
        "/home/msai/ghu003/6127project_experiments/outputs/all_datasets_strict_embeddings.json").resolve()
    model_path = "/home/msai/ghu003/Qwen2.5-7B-Instruct/"
    eval_data = ["medqa", "medmcqa", "pubmedqa"]
    num_samples = None

    # 加载模型和 tokenizer
    print(f"📦 加载模型中：{model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    print("✅ 模型加载成功！")

    # 遍历评估数据集
    for data in eval_data:
        if data == "medqa":
            dataset = load_dataset("bigbio/med_qa", "med_qa_en_4options_source", trust_remote_code=True)['test']
        elif data == "medmcqa":
            dataset = load_dataset("medmcqa")['validation'].filter(lambda x: x['cop'] in [0, 1, 2, 3])
        elif data == "pubmedqa":
            dataset = load_dataset("pubmed_qa", "pqa_labeled")['train']
        else:
            continue

        # ✅ 显示第一条 Prompt 与示例
        show_first_prompt_and_examples(dataset, data, embedding_file)
        break  # 只展示一个即可调试，如需全量评估再删掉这一行

if __name__ == "__main__":
    main()
