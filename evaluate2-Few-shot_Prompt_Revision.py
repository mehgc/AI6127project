# Load model directly
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

# Formatting functions for each dataset
def format_medqa(example):
    question = example['question']
    options_str = "\n".join([f"{opt['key']}: {opt['value']}" for opt in example['options']])
    correct_answer_idx = example['answer_idx']  # e.g., "A", "B", "C", "D"
    prompt = f"""Below is a medical question with multiple choice options. Provide only the letter of the correct answer (e.g., A, B, C, D).

### Question:
{question}

### Options:
{options_str}

### Answer:
{correct_answer_idx}"""
    return {"text": prompt, "answer_idx": correct_answer_idx}


def format_medmcqa(example):
    # Map integer cop to letter (0 -> A, 1 -> B, 2 -> C, 3 -> D)
    cop_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    if example['cop'] not in [0, 1, 2, 3]:  # Skip invalid samples
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
    # Convert yes/no/maybe to A, B, C options
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
    """
    根据数据集类型，调用对应的格式化函数，返回标准化 input prompt。
    用于 inference 阶段构造 query。
    """
    if dataset_type == "medqa":
        return format_medqa(example)["text"]
    elif dataset_type == "medmcqa":
        formatted = format_medmcqa(example)
        return formatted["text"] if formatted else None
    elif dataset_type == "pubmedqa":
        return format_pubmedqa(example)["text"]
    else:
        raise ValueError(f"❌ 不支持的数据集类型: {dataset_type}")


def evaluate_model_with_dynamic_fewshot(model, tokenizer, test_dataset, dataset_type, embedding_file, num_samples=None):

    predictions = []
    ground_truth = []

    if num_samples:
        test_samples = test_dataset.select(range(min(num_samples, len(test_dataset))))
    else:
        test_samples = test_dataset

    for example in tqdm(test_samples, desc=f"Evaluating {dataset_type}", ncols=80):
        # ✅ 使用统一格式化入口
        query = get_query_prompt_without_answer(example, dataset_type)
        if query is None:
            continue  # 跳过格式不合法样本

        # ✅ 获取标准答案
        if dataset_type == "medqa":
            correct_answer = example["answer_idx"]
        elif dataset_type == "medmcqa":
            correct_answer = {0: "A", 1: "B", 2: "C", 3: "D"}[example["cop"]]
        elif dataset_type == "pubmedqa":
            final = example["final_decision"].lower()
            if final not in {"yes", "no", "maybe"}:
                continue
            correct_answer = {"yes": "A", "no": "B", "maybe": "C"}[final]
        else:
            raise ValueError(f"不支持的数据集类型：{dataset_type}")

        # ✅ 构建 Few-shot Prompt
        top_examples = find_top_k_similar_examples(query, embedding_file)
        full_prompt = construct_prompt(top_examples, query)

        # ✅ 模型生成
        device = model.device
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        # ==⚠️⚠️⚠️⚠️⚠️⚠️⚠️【需要调整】⚠️⚠️⚠️⚠️⚠️⚠️⚠️ =====
        outputs = model.generate(
            **inputs,
            max_new_tokens=2,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,  # ✅ 禁用采样，保证确定性
            temperature=0.0  # ✅ 输出更稳定
        )
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # ✅ 答案提取
        predicted = "X"
        for ch in generated_text:
            if ch in ["A", "B", "C", "D"]:
                predicted = ch
                break

        predictions.append(predicted)
        ground_truth.append(correct_answer)

    # ✅ 指标计算
    acc = accuracy_score(ground_truth, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average='weighted', zero_division=0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }, predictions, ground_truth


def main():
    # ✅ 参数配置
    embedding_file = Path("/home/msai/ghu003/6127project_experiments/outputs/all_datasets_strict_embeddings.json").resolve()
    num_samples = 600
    eval_data = ["medqa", "medmcqa", "pubmedqa"]

    # ✅ 模型路径（直接定位到 snapshots 目录）
    model_path = "/home/msai/ghu003/LLaMA-Factory/models/qwen_medqa_medmcqa_pubmedqa_lora_v2/"


    # ✅ 加载模型和 tokenizer
    try:
        print(f"📦 加载模型中：{model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16  # ✅ 推荐加上，加速推理（若支持）
        )
        print("✅ 模型加载成功，准备开始评估任务！")
    except Exception as e:
        print(f"❌ 模型加载失败！错误信息：\n{e}")
        return


    # ✅ 遍历评估数据集
    for data in eval_data:
        if data == "medqa":
            dataset = load_dataset("bigbio/med_qa", "med_qa_en_4options_source", trust_remote_code=True)
            test_set = dataset['test']
        elif data == "medmcqa":
            dataset = load_dataset("medmcqa")
            test_set = dataset['validation'].filter(lambda x: x['cop'] in [0, 1, 2, 3])
        elif data == "pubmedqa":
            dataset = load_dataset("pubmed_qa", "pqa_labeled")
            test_set = dataset['train']
        else:
            continue

        print(f"\n--- Evaluating on {data.upper()} ---")
        metrics, preds, gts = evaluate_model_with_dynamic_fewshot(
            model, tokenizer, test_set, data, embedding_file, num_samples=num_samples
        )
        print(f"Metrics: {metrics}")
        print(f"Sample Predictions: {preds[:5]}")
        print(f"Sample Ground Truth: {gts[:5]}")


if __name__ == '__main__':
    main()