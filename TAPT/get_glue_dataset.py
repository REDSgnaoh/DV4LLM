from datasets import load_dataset
import json

# Load only the "train" split of the dataset
file_name = "mnli"
dataset = load_dataset("glue", file_name, split="train")


output_jsonl_file = 'glue_dataset/mnli/mnli.jsonl'

with open(output_jsonl_file, 'w') as jsonl_file:
    for id, text1, text2 in zip(dataset.data['idx'], dataset.data['premise'], dataset.data['hypothesis']):
        text = text1.as_py() + ' ' + text2.as_py()
        json_item = json.dumps({
                    "id": id.as_py(),
                    "text": text
                })
        
        jsonl_file.write(json_item + '\n')





