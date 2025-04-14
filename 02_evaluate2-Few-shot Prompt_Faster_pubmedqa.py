# Load model directly
import os
import json
import torch
import time
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from tqdm import tqdm

# ✅ 提前加载 SentenceTransformer 模型（仅加载一次）
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# 嵌入文件缓存：避免重复读取 json
embedding_cache = {}

# ✅ 向量文件缓存函数
def load_embedding_file(embedding_file_path):
    path = str(embedding_file_path)
    if path not in embedding_cache:
        print(f"🧠 加载嵌入文件：{path}")
        with open(path, "r", encoding="utf-8") as f:
            embedding_cache[path] = json.load(f)
    return embedding_cache[path]

# ✅ 找出最相似的 top-k 示例
def find_top_k_similar_examples(query, embedding_file, top_k=3):
    data = load_embedding_file(embedding_file)  # ✅ 使用缓存

    # ✅ 获取嵌入列表
    all_embeddings = [item["embedding"] for item in data]
    query_embedding = similarity_model.encode(query)

    scores = util.cos_sim(query_embedding, all_embeddings)[0]
    top_results = scores.topk(k=top_k)

    similar_examples = []
    for score, idx in zip(top_results.values, top_results.indices):
        matched = data[idx]
        similar_examples.append({
            "input": matched.get("input_short", ""),  # ✅ 从 input_short 取字段
            "output": matched["output"]
        })
    return similar_examples


# ✅ 批量推理函数 batch_generate()
def batch_generate(prompts: list, tokenizer, model, max_new_tokens=2):
    """
    批量推理：输入多个 prompt，统一 tokenizer 编码 + 模型生成 + 解码预测答案（A/B/C/D）
    带耗时统计与 token 长度警告
    """
    import time
    start_tok = time.time()

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    end_tok = time.time()
    encode_time = end_tok - start_tok
    if encode_time > 1.0:
        tqdm.write(f"⏱️ tokenizer 编码耗时：{encode_time:.2f}s （batch_size={len(prompts)}）")

    # ⚠️ 检查 token 长度（可视化提示）
    token_lens = [len(tokenizer(p)["input_ids"]) for p in prompts]
    max_len = max(token_lens)
    if max_len > 1800:
        tqdm.write(f"⚠️ 警告：本批最大 prompt 长度接近限制：{max_len} tokens")

    # 🔮 模型生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    # 🔍 提取预测结果
    results = []
    for i in range(len(prompts)):
        input_len = (inputs.input_ids[i] != tokenizer.pad_token_id).sum().item()
        output_text = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True).strip()

        predicted = "X"
        for ch in output_text:
            if ch in ["A", "B", "C", "D"]:
                predicted = ch
                break
        results.append(predicted)

    return results


# ✅ 构造 Few-shot Prompt
def construct_prompt(examples, query, verbose=False):
    """
    构建 Few-shot Prompt（含多个示例 + 当前 query）
    """
    if verbose:
        tqdm.write(f"🧱 正在构造 Prompt（含 {len(examples)} 个示例）")

    prompt = ""
    for ex in examples:
        prompt += f"{ex['input']}\n\n### Answer:\n{ex['output']}\n\n---\n\n"
    prompt += query + "\n\n### Let's think step by step:\n"
    return prompt


# Formatting functions for each dataset
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


def evaluate_model_with_dynamic_fewshot(model, tokenizer, test_dataset, embedding_file, num_samples=None, batch_size=2):
    predictions = []
    ground_truth = []

    if num_samples:
        test_samples = test_dataset.select(range(min(num_samples, len(test_dataset))))
    else:
        test_samples = test_dataset

    print(f"📊 开始评估 PubMedQA，共 {len(test_samples)} 条样本，batch size = {batch_size}")

    # ✅ 计算总 batch 数 + 创建 tqdm 外层
    total_batches = (len(test_samples) + batch_size - 1) // batch_size
    batch_iter = range(0, len(test_samples), batch_size)

    for i in tqdm(batch_iter, desc=f"[PubMedQA] 推理中...", total=total_batches, ncols=80):
        batch = test_samples[i:i+batch_size]
        batch_prompts = []
        batch_labels = []

        # ✅ 每个 query 检索示例加小进度条
        for example in tqdm(batch, desc="🔍 检索中", leave=False, ncols=70):
            query = format_pubmedqa(example)["text"]
            if query is None:
                continue

            # ✅ 标准答案
            final = example["final_decision"].lower()
            if final not in {"yes", "no", "maybe"}:
                continue
            correct_answer = {"yes": "A", "no": "B", "maybe": "C"}[final]

            # ✅ 构建 prompt（含 top-k 检索）
            top_examples = find_top_k_similar_examples(query, embedding_file)
            full_prompt = construct_prompt(top_examples, query, verbose=True)

            batch_prompts.append(full_prompt)
            batch_labels.append(correct_answer)

        # ✅ 批量推理
        if batch_prompts:
            batch_preds = batch_generate(batch_prompts, tokenizer, model)
            predictions.extend(batch_preds)
            ground_truth.extend(batch_labels)

    # ✅ 评估
    acc = accuracy_score(ground_truth, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average='weighted', zero_division=0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }, predictions, ground_truth



def main():
    embedding_file = Path("/home/msai/ghu003/6127project_experiments/outputs/emb_pubmedqa.json").resolve()
    batch_size = 8  # ✅ 可根据显存调整
    num_samples = 500  # ✅ 可设为具体数字调试

    model_id = "/home/msai/ghu003/Qwen2.5-7B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        print(f"📦 正在加载在线模型：{model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        model.eval()
        print("✅ 模型加载成功，准备开始评估 PubMedQA")
    except Exception as e:
        print(f"❌ 模型加载失败！错误信息：\n{e}")
        return

    # ✅ 加载 PubMedQA 测试集
    dataset = load_dataset("pubmed_qa", "pqa_labeled")
    test_set = dataset["test"]

    print(f"\n🌐 使用嵌入库：{embedding_file.name}")
    print(f"📊 测试集样本数：{len(test_set)}")

    start_time = time.time()

    metrics, preds, gts = evaluate_model_with_dynamic_fewshot(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_set,
        embedding_file=embedding_file,
        num_samples=num_samples,
        batch_size=batch_size
    )

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"📊 PubMedQA 评估结果：{metrics}")
    print(f"📌 示例预测：{preds[:5]}")
    print(f"📌 示例答案：{gts[:5]}")
    print(f"🕒 总用时：{elapsed:.2f} 秒")



if __name__ == "__main__":
    main()
