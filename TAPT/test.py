# from allennlp.data.tokenizers import WordTokenizer
# from allennlp.data import Vocabulary
# from allennlp.data.token_indexers import SingleIdTokenIndexer

# # Step 2: Create a tokenizer
# tokenizer = WordTokenizer()

# # Step 3: Create a vocabulary
# # For simplicity, we manually create a vocabulary here.
# # In a real scenario, you would create a vocabulary from your dataset.
# vocab = Vocabulary(tokens_to_add={"tokens": ["<PAD>", "<UNK>", "I", "love", "Allennlp"]})

# # Step 4: Create a bag-of-words representation
# # Tokenize the text
# tokens = tokenizer.tokenize("I love Allennlp")

# # Create a token indexer to convert tokens to their IDs
# token_indexer = SingleIdTokenIndexer(namespace='tokens')
# token_indices = token_indexer.tokens_to_indices(tokens, vocab, index_name='tokens')

# # Get the token IDs
# token_ids = token_indices["tokens"]

# # Create a bag-of-words representation
# bow_representation = [0] * vocab.get_vocab_size("tokens")
# for token_id in token_ids:
#     bow_representation[token_id] += 1

# print(bow_representation)
# print("token_ids:", token_ids)
# print("token_indices:", token_indices)

# import torch

# # Create a one-dimensional tensor
# one_dimensional_tensor = torch.tensor([1, 2, 3, 4, 5])

# # Use torch.unsqueeze to add a new dimension along axis 0 (rows)
# two_dimensional_tensor = torch.unsqueeze(one_dimensional_tensor, 1)

# print(two_dimensional_tensor)

import numpy as np

# Load the npz file
data = np.load('Embeddings/cs_sci/embedding.npz', allow_pickle=True)

# List all arrays in the file
print(data.files)

# Access the arrays by their respective keys
ids_ = data['ids_']
vecs_ = data['vecs_']

# Print or inspect the loaded data
print(ids_.shape)
print(vecs_.shape)


# import pickle
# with open('train_data/news_full.pkl', 'rb') as file:
#     data = pickle.load(file)

# print(data)

# import numpy as np

# # Create a sample array with shape (1, 81)
# arr = np.array([[1, 2, 3, 80, 81]])  # This array has 81 elements
# print(arr.shape[1])
# # Use the reshape method to change it to shape (81)
# arr_reshaped = arr.reshape(arr.shape[1])
# print(arr_reshaped.shape)

# import numpy as np
# import pickle

# data = np.load('selected_data/cola/Cola_Vampire_Tapt', allow_pickle=True)

# # List all arrays in the file
# print(len(data))