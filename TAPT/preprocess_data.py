import json
import os
from typing import List

import nltk
import numpy as np
import pandas as pd
import spacy
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.tokenizer import Tokenizer
from tqdm import tqdm

from util import read_text, save_sparse, write_to_json


def load_data(data_path: str, tokenize: bool = False, tokenizer_type: str = "just_spaces") -> List[str]:
    if tokenizer_type == "just_spaces":
        tokenizer = SpacyWordSplitter()
    elif tokenizer_type == "spacy":
        nlp = spacy.load('en')
        tokenizer = Tokenizer(nlp.vocab)
    tokenized_examples = []
    with tqdm(open(data_path, "r"), desc=f"loading {data_path}") as f:
        for line in f:
            if data_path.endswith(".jsonl") or data_path.endswith(".json"):
                example = json.loads(line)
            else:
                example = {"text": line}
            if tokenize:
                if tokenizer_type == 'just_spaces':
                    tokens = list(map(str, tokenizer.split_words(example['text'])))
                elif tokenizer_type == 'spacy':
                    tokens = list(map(str, tokenizer(example['text'])))
                text = ' '.join(tokens)
            else:
                text = example['text']
            tokenized_examples.append(text)
    return tokenized_examples

def write_list_to_file(ls, save_path):
    """
    Write each json object in 'jsons' as its own line in the file designated by 'save_path'.
    """
    # Open in appendation mode given that this function may be called multiple
    # times on the same file (positive and negative sentiment are in separate
    # directories).
    out_file = open(save_path, "w+")
    for example in ls:
        out_file.write(example)
        out_file.write('\n')


def main():
    # Define your arguments here
    train_path = "train_data/arxiv_train.jsonl"  # replace with your train path
    dev_path = "train_data/arxiv_dev.jsonl"  # replace with your dev path
    serialization_dir = "data/arxiv"  # replace with your serialization directory
    tfidf = True  # or False
    vocab_size = 30000  # replace with your vocab size
    tokenize = True  # or False
    tokenizer_type = "just_spaces"  # replace with your tokenizer type
    reference_corpus_path = ""  # replace with your reference corpus path, or None if not using
    tokenize_reference = True  # or False
    reference_tokenizer_type = "just_spaces"  # replace with your reference tokenizer type

    if not os.path.isdir(serialization_dir):
        os.mkdir(serialization_dir)
    
    vocabulary_dir = os.path.join(serialization_dir, "vocabulary")

    if not os.path.isdir(vocabulary_dir):
        os.mkdir(vocabulary_dir)

    tokenized_train_examples = load_data(train_path, tokenize, tokenizer_type)
    tokenized_dev_examples = load_data(dev_path, tokenize, tokenizer_type)

    print("fitting count vectorizer...")
    if tfidf:
        count_vectorizer = TfidfVectorizer(stop_words='english', max_features=vocab_size, token_pattern=r'\b[^\d\W]{3,30}\b')
    else:
        count_vectorizer = CountVectorizer(stop_words='english', max_features=vocab_size, token_pattern=r'\b[^\d\W]{3,30}\b')
    
    text = tokenized_train_examples + tokenized_dev_examples
    
    count_vectorizer.fit(tqdm(text))

    vectorized_train_examples = count_vectorizer.transform(tqdm(tokenized_train_examples))
    vectorized_dev_examples = count_vectorizer.transform(tqdm(tokenized_dev_examples))

    if tfidf:
        reference_vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b[^\d\W]{3,30}\b')
    else:
        reference_vectorizer = CountVectorizer(stop_words='english', token_pattern=r'\b[^\d\W]{3,30}\b')
    if not reference_corpus_path:
        print("fitting reference corpus using development data...")
        reference_matrix = reference_vectorizer.fit_transform(tqdm(tokenized_dev_examples))
    else:
        print(f"loading reference corpus at {reference_corpus_path}...")
        reference_examples = load_data(reference_corpus_path, tokenize_reference, reference_tokenizer_type)
        print("fitting reference corpus...")
        reference_matrix = reference_vectorizer.fit_transform(tqdm(reference_examples))

    reference_vocabulary = reference_vectorizer.get_feature_names()

    vectorized_train_examples = sparse.hstack((np.array([0] * len(tokenized_train_examples))[:,None], vectorized_train_examples))
    vectorized_dev_examples = sparse.hstack((np.array([0] * len(tokenized_dev_examples))[:,None], vectorized_dev_examples))
    master = sparse.vstack([vectorized_train_examples, vectorized_dev_examples])

    # generate background frequency
    print("generating background frequency...")
    bgfreq = dict(zip(count_vectorizer.get_feature_names(), (np.array(master.sum(0)) / vocab_size).squeeze()))

    print("saving data...")
    save_sparse(vectorized_train_examples, os.path.join(serialization_dir, "train.npz"))
    save_sparse(vectorized_dev_examples, os.path.join(serialization_dir, "dev.npz"))
    if not os.path.isdir(os.path.join(serialization_dir, "reference")):
        os.mkdir(os.path.join(serialization_dir, "reference"))
    save_sparse(reference_matrix, os.path.join(serialization_dir, "reference", "ref.npz"))
    write_to_json(reference_vocabulary, os.path.join(serialization_dir, "reference", "ref.vocab.json"))
    write_to_json(bgfreq, os.path.join(serialization_dir, "vampire.bgfreq"))
    
    write_list_to_file(['@@UNKNOWN@@'] + count_vectorizer.get_feature_names(), os.path.join(vocabulary_dir, "vampire.txt"))
    write_list_to_file(['*tags', '*labels', 'vampire'], os.path.join(vocabulary_dir, "non_padded_namespaces.txt"))

if __name__ == '__main__':
    main()
