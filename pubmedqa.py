import random

from datasets import load_dataset
import os
import json

# Function to save data as JSON with specified columns
def save_as_json(data, filename, max_items=8000):
    file_path = os.path.join(save_path, filename)
    cop_to_letter = {'yes': "A", 'no': "B", 'maybe': "C"}

    # First filter all valid items
    valid_items = [
        item for item in data
        if item['final_decision'] in ['yes', 'no', 'maybe']
    ]

    # Then take a random sample (or all if there are fewer than max_items)
    sampled_items = random.sample(
        valid_items,
        min(max_items, len(valid_items))
    )

    data_to_save = []
    for item in sampled_items:
        choice = cop_to_letter[item['final_decision']]
        data_to_save.append({
            "instruction": "Below is a medical question with multiple choice options. Provide only the letter of the correct answer (e.g., A, B, C).",
            "input": f"""### Question:
{item['question']}

### Options:
A: Yes\nB: No\nC: Maybe

### Answer:""",
            "output": f"{choice}",
        })

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset("pubmed_qa", "pqa_artificial")

    # Define the save path
    save_path = "./"
    os.makedirs(save_path, exist_ok=True)

    # Save the modified data for train, validation, and test splits
    save_as_json(dataset['train'], 'pubmedqa_train.json')



