import json
import random

input_files = ['train_data/arxiv.jsonl']

training_files = ['train_data/arxiv_train.jsonl']

dev_files = ['train_data/arxiv_dev.jsonl']

split_ratio = 0.9

for input_file, training_file, dev_file in zip(input_files, training_files, dev_files):
    with open(input_file, 'r', encoding='utf-8') as infile, \
        open(training_file, 'w', encoding='utf-8') as train_outfile, \
        open(dev_file, 'w', encoding='utf-8') as dev_outfile:

        for line in infile:
            if random.random() < split_ratio:
                train_outfile.write(line)
            else:
                dev_outfile.write(line)
        
infile.close()
train_outfile.close()
dev_outfile.close()


