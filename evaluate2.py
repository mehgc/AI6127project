import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# Formatting functions for each dataset
def format_medqa_prompt(example):
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

def format_medmcqa_prompt(example):
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

def format_pubmedqa_prompt(example):
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
    question = example['question']
    if dataset_type == "medqa":
        options_str = "\n".join([f"{opt['key']}: {opt['value']}" for opt in example['options']])
    elif dataset_type == "medmcqa":
        options_str = f"A: {example['opa']}\nB: {example['opb']}\nC: {example['opc']}\nD: {example['opd']}"
    elif dataset_type == "pubmedqa":
        options_str = "A: Yes\nB: No\nC: Maybe"
        prompt = f"""Below is a medical question with multiple choice options. Provide only the letter of the correct answer (e.g., A, B, C)

### Question:
{question}

### Options:
{options_str}

### Answer:"""
        return prompt
    prompt = f"""Below is a medical question with multiple choice options. Provide only the letter of the correct answer (e.g., A, B, C, D).

### Question:
{question}

### Options:
{options_str}

### Answer:"""
    return prompt

def evaluate_model(model, tokenizer, test_dataset, dataset_type, num_samples=None):
    predictions = []
    ground_truth = []

    # Use .select() to safely slice the dataset
    if num_samples is not None:
        test_samples = test_dataset.select(range(min(num_samples, len(test_dataset))))
    else:
        test_samples = test_dataset

    for example in test_samples:
        inference_prompt_text = format_inference_prompt(example, dataset_type)
        if dataset_type == "medqa":
            correct_answer_idx = example['answer_idx']
        elif dataset_type == "medmcqa":
            if example['cop'] not in [0, 1, 2, 3]:  # Skip invalid samples
                continue
            cop_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
            correct_answer_idx = cop_to_letter[example['cop']]
        elif dataset_type == "pubmedqa":
            answer_map = {"yes": "A", "no": "B", "maybe": "C"}
            correct_answer_idx = answer_map[example['final_decision'].lower()]

        # Tokenize input and generate output
        inputs = tokenizer(inference_prompt_text, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=2, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Extract the predicted answer (assume it's a single letter)
        predicted_answer = generated_text[0] if generated_text else "X"  # Default to "X" if empty

        predictions.append(predicted_answer)
        ground_truth.append(correct_answer_idx)

    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average='weighted',
                                                               zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }, predictions, ground_truth


def main():
    num_samples = 1000
    model_name = "/home/msai/ghu003/LLaMA-Factory/models/qwen_medqa_medmcqa_pubmedqa_lora_v2/"
    eval_data = [
                 "medqa",
                 "medmcqa",
                 "pubmedqa",
                 ]

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    model.eval()

    print(f"Evaluating model {model_name}")

    for data in eval_data:
        if data == "medqa":
            # MedQA
            medqa_dataset = load_dataset("bigbio/med_qa", "med_qa_en_4options_source", trust_remote_code=True)
            medqa_test = medqa_dataset['test']

            print("\n--- Evaluating on MedQA Test ---")
            metrics_medqa, preds_medqa, gt_medqa = evaluate_model(model, tokenizer, medqa_test, "medqa",
                                                                  num_samples=num_samples)
            print(f"Metrics for MedQA Test: {metrics_medqa}")
            print(f"Sample Predictions: {preds_medqa[:5]}")
            print(f"Sample Ground Truth: {gt_medqa[:5]}")

        elif data == "medmcqa":
            # MedMCQA
            medmcqa_dataset = load_dataset("medmcqa")
            medmcqa_test = medmcqa_dataset['validation']
            # Filter out invalid samples (cop = -1)
            medmcqa_test = medmcqa_test.filter(lambda x: x['cop'] in [0, 1, 2, 3])

            print("\n--- Evaluating on MedMCQA Test ---")
            metrics_medmcqa, preds_medmcqa, gt_medmcqa = evaluate_model(model, tokenizer, medmcqa_test, "medmcqa",
                                                                        num_samples=num_samples)
            print(f"Metrics for MedMCQA Test: {metrics_medmcqa}")
            print(f"Sample Predictions: {preds_medmcqa[:5]}")
            print(f"Sample Ground Truth: {gt_medmcqa[:5]}")

        elif data == "pubmedqa":
            # PubMedQA
            pubmedqa_dataset = load_dataset("pubmed_qa", "pqa_labeled")
            pubmedqa_train = pubmedqa_dataset['train']

            print("\n--- Evaluating on PubMedQA Train ---")
            metrics_pubmedqa, preds_pubmedqa, gt_pubmedqa = evaluate_model(model, tokenizer, pubmedqa_train, "pubmedqa",
                                                                           num_samples=num_samples)
            print(f"Metrics for PubMedQA Train: {metrics_pubmedqa}")
            print(f"Sample Predictions: {preds_pubmedqa[:5]}")
            print(f"Sample Ground Truth: {gt_pubmedqa[:5]}")

if __name__ == '__main__':
    main()