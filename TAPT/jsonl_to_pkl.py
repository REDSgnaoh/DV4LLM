import json
import pickle
import numpy as np




jsonl_file_paths = ['selected_data/cs_sci/50k.jsonl', 'selected_data/cs_sci/150k.jsonl']
                   

# jsonl_file_paths = ['Target_datasets/reviews_imdb_dev.jsonl', 'Target_datasets/reviews_imdb_train.jsonl',
#                     'Target_datasets/reviews_imdb_test.jsonl']

for jsonl_file_path in jsonl_file_paths:
    # List to store the data from the current JSONL file
    data_list = []

    # Read data from the current JSONL file
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)

    # Define the output pkl file path for the current JSONL file
    output_pkl_file = f'{jsonl_file_path}.pkl'

    # Store the data from the current JSONL file in its corresponding pkl file
    with open(output_pkl_file, 'wb') as pkl_file:
        pickle.dump(data_list, pkl_file)

    print(f'Data from {jsonl_file_path} stored in {output_pkl_file}')


# data = np.load('Target_datasets/reviews_imdb_dev.jsonl.pkl', allow_pickle=True)
# print(data[0])
