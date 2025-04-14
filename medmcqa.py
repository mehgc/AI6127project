from datasets import load_dataset
import os
import json


# Function to save data as JSON with specified columns
def save_as_json(data, filename):
    file_path = os.path.join(save_path, filename)
    data_to_save = []
    cop_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    i = 0
    max_sample = 10000

    # Modify the data to include only 'question' and 'answer' columns
    for item in data:
        i += 1
        if i > max_sample:
            break
        if item['cop'] not in [0, 1, 2, 3]:  # Skip invalid samples
            continue
        choice = cop_to_letter[item['cop']]
        data_to_save.append({
            "instruction": "Below is a medical question with multiple choice options. Provide only the letter of the correct answer (e.g., A, B, C, D).",
            "input": f"""### Question:
{item['question']}

### Options:
A: {item['opa']}\nB: {item['opb']}\nC: {item['opc']}\nD: {item['opd']}

### Answer:""",
            "output": f"{choice}",
        })

    # Write the modified data to a JSON file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset("medmcqa")

    # Define the save path
    save_path = "./"
    os.makedirs(save_path, exist_ok=True)

    # Save the modified data for train, validation, and test splits
    save_as_json(dataset['train'], 'medmcqa_train.json')



