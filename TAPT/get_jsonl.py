import pickle
import json
import os
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    


filepaths = ['train_data/arxiv.pkl']
output_jsonl_files = ['train_data/arxiv.jsonl']


for filepath, output_jsonl_file in zip(filepaths, output_jsonl_files):
    # Get the base file name (without extension) to use for the .json file name
    index = 1
    
    # Step 2: Load the .pkl file
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
        
    with open(output_jsonl_file, 'w') as jsonl_file:
        for item in data:
            # Serialize the item to JSON
            json_item = json.dumps({
                "id": index,
                "text": item
            })
            # Write the JSON item to the .jsonl file
            jsonl_file.write(json_item + '\n')
            index += 1
            if(index >= 1000000):
                break


# filepaths = [
#              'Target_datasets/cs_citation_dev.jsonl', 'Target_datasets/cs_scierc_dev.jsonl']
# output_jsonl_files = [
#                       'Target_datasets/cs_citation_id_dev.jsonl', 'Target_datasets/cs_scierc_id_dev.jsonl']


# for filepath, output_jsonl_file in zip(filepaths, output_jsonl_files):
#     # Get the base file name (without extension) to use for the .json file name
#     index = 1
    
#     # Step 2: Load the .pkl file
#     with open(filepath, 'rb') as file, open(output_jsonl_file, 'w') as jsonl_file:
#         for line in file:
#             # Serialize the item to JSON
#             data = json.loads(line)
#             json_item = json.dumps({
#                 "id": index,
#                 "text": data['text']
#             })
#             # Write the JSON item to the .jsonl file
#             jsonl_file.write(json_item + '\n')
#             index += 1
#             if(index >= 1000000):
#                 break

