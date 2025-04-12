from datasets import load_dataset
import os
import json


# Function to save data as JSON with specified columns
def save_as_json(data, filename):
    file_path = os.path.join(save_path, filename)
    data_to_save = []

    # Modify the data to include only 'question' and 'answer' columns
    for item in data:
        data_to_save.append({
            "instruction": "Below is a medical question with multiple choice options. Provide only the letter of the correct answer (e.g., A, B, C, D).",
            "input": f"""### Question:
{item['question']}

### Options:
{item['options'][0]['key']}: {item['options'][0]['value']}
{item['options'][1]['key']}: {item['options'][1]['value']}
{item['options'][2]['key']}: {item['options'][2]['value']}
{item['options'][3]['key']}: {item['options'][3]['value']}

### Answer:""",
            "output": item['answer_idx'],
        })

    # Write the modified data to a JSON file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset("bigbio/med_qa", "med_qa_en_4options_source")

    # Define the save path
    save_path = "./"
    os.makedirs(save_path, exist_ok=True)

    # Save the modified data for train, validation, and test splits
    save_as_json(dataset['train'], 'med_qa_train.json')
    # save_as_json(dataset['test'], 'med_qa_test.json')



