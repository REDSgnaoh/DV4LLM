import numpy as np
from sklearn.neighbors import NearestNeighbors
import json

# Load the embeddings from the first file
embeddings_file_1 = np.load('Embeddings/mrpc/embedding.npz', allow_pickle=True)
ids_glue = embeddings_file_1['ids_']
vecs_glue = embeddings_file_1['vecs_']

# Load the embeddings from the second file
embeddings_file_2 = np.load('Embeddings/pubmed/embedding.npz', allow_pickle=True)
ids_train = embeddings_file_2['ids_']
vecs_train = embeddings_file_2['vecs_']

if(ids_glue.shape[0] > 40000):
    ids_glue = ids_glue[:40000]
    vecs_glue = vecs_glue[:40000]

selected_30k = []
selected_40k = []


# Initialize a NearestNeighbors model using the embeddings from the second file and cosine distance
k_neighbors = 11  # You can change this to your desired value of k
nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine').fit(vecs_train)

# Find the k-nearest neighbors of the embeddings from the first file in the space of the second file
_, indices = nbrs.kneighbors(vecs_glue)

# Create a dictionary to store the nearest neighbors for each point in the smaller file
nearest_neighbors_list = []

# Iterate through the indices and store the nearest neighbors with their IDs
for i, neighbors in enumerate(indices):
    nearest_neighbors_list.append([ids_train[neighbor_idx] for _, neighbor_idx in enumerate(neighbors)]) 

selected_ids = np.array(nearest_neighbors_list)
print("selected size:", selected_ids.shape)


# get the selected ids for different task size
basic_samples = selected_ids.shape[0]
print("numbers for one loop:", basic_samples)

flag_30k = 0
flag_40k = 0

loop_numbers = 0
while (flag_40k < 40000):
    for i in range(basic_samples):
        if flag_30k < 30000:
            selected_30k.append(selected_ids[i][loop_numbers])
            flag_30k = flag_30k + 1

        if flag_40k < 40000:
            selected_40k.append(selected_ids[i][loop_numbers])
            flag_40k = flag_40k + 1

            
    loop_numbers = loop_numbers + 1

print("finished select ids!")
print("flags:", flag_30k)
print("flags:", flag_40k)




from collections import Counter

# Function to read a JSONL file and select data based on specified IDs
def select_data_by_ids(file_path, target_ids):
    selected_data = []
    id_counter = {}

    # Create a dictionary to count occurrences of each target ID
    for id in target_ids:
        id_counter[id] = id_counter.get(id, 0) + 1

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            obj_id = json_obj.get("id")

            while (obj_id in id_counter and id_counter[obj_id]) > 0:
                selected_data.append(json_obj)
                id_counter[obj_id] -= 1
    
    print("selected_size:", len(selected_data))
    return selected_data


def write_selected_data_to_jsonl(selected_data, output_file_path):
    with open(output_file_path, 'w') as output_file:
        for data in selected_data:
            json.dump(data, output_file)
            output_file.write('\n')


# Input and output file paths
domain_file_path = 'train_data/pubmed.jsonl'
selected_file_paths = ['selected_data/mrpc/30k.jsonl', 'selected_data/mrpc/40k.jsonl']

ids_to_selects = [selected_30k, selected_40k]

for output_file_path, ids_to_select in zip(selected_file_paths, ids_to_selects):
    # Call the function to select the data
    selected_data = select_data_by_ids(domain_file_path, ids_to_select)

    # Call the function to write the selected data to a new JSONL file
    write_selected_data_to_jsonl(selected_data, output_file_path)

    print("Selected data has been written to", output_file_path)
    