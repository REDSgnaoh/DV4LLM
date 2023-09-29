import numpy as np
import pandas as pd 

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import pickle
import lzma

#ARGPARSE
from datasets import Dataset
import pandas as pd
from datasets import load_dataset

import argparse

parser = argparse.ArgumentParser()
# add_dataset_model_arguments(parser)

parser.add_argument('--model_name', type=str, required=True,
                    help='model name (bert base cased/uncased)')

parser.add_argument('--task_name', type=str, required=True,
                    help='name of the task')

parser.add_argument('--data_size', type=int, required=True,
                    help='dataset size')

parser.add_argument('--method', type=str, required=True,
                    help='name of the method (rand/ot/dsir/tapt/prand/corrot)')

parser.add_argument('--learning_rate', type=float, required=False,
                    help='learning rate', default=2e-5)

parser.add_argument('--epochs', type=int, required=False,
                    help='number of epochs', default=5)

parser.add_argument('--run', type=int, required=True,
                    help='The run number', default=5)

parser.add_argument('--batch_size', type=int, required=False,
                    help='batch size', default=16)

parser.add_argument('--max_length', type=int, required=False,
                    help='max length', default=128)

arg = parser.parse_args() # args conflict with other argument

print(f"model_name {arg.model_name}")

print(f"dataset size {arg.data_size}")

print(f"task name {arg.task_name}")

print(f"method name {arg.method}")

print(f"learning rate {arg.learning_rate}")

print(f"epochs {arg.epochs}")

print(f"run num {arg.run}")

print(f"batch size {arg.batch_size}")

print(f"max length {arg.max_length}")

torch.cuda.current_device()

# Define constants for MLM fine-tuning
model_name = arg.model_name
batch_size = arg.batch_size
epochs_mlm = arg.epochs
learning_rate_mlm = arg.learning_rate
folder = 'data_ot/'
max_length = arg.max_length
data_size = arg.data_size
task_name = arg.task_name
method = arg.method
run = arg.run
seed = 0

print("seed: ", seed)


if data_size == 50000:
    if task_name in ['rte', 'mrpc']:
        if model_name == 'bert-base-uncased':
            model_name = "ishan/bert-base-uncased-mnli"
        elif model_name == 'bert-base-cased':
            model_name = "WillHeld/bert-base-cased-mnli"
        elif model_name == "roberta-base":
            model_name = "textattack/roberta-base-MNLI"
    
# Load and preprocess your data for MLM fine-tuning
if method == "tapt":
    data_size_k = int(data_size/1000)
    dataset_name = f"tapt/{task_name}/{data_size_k}k.jsonl.pkl"
    data_path_mlm = folder + dataset_name  # Update with your data path
    
    # dataset = Dataset.from_dict(t)
    file = pickle.load(open(data_path_mlm, "rb"))
    t_dict = Dataset.from_pandas(pd.DataFrame(data=file))
    data_mlm = t_dict['text']
elif method == "dsir":
    dataset_name = f"{task_name}/retrieved_50000_nopack.json"
    data_path_mlm = folder + dataset_name
    file = load_dataset('json', data_files=data_path_mlm)
    data_mlm = file['train']['text']
elif method == "ot":
    dataset_name = f"ot_8/{task_name}_ot_2m_50k.pkl"
    data_path_mlm = folder + dataset_name
    with open(data_path_mlm, 'rb') as file:
        data_mlm = pickle.load(file)
    data_mlm=np.array(data_mlm).tolist()
elif method == "prand":
    dataset_name = f"prand/permRnd50k{run}.pkl"
    data_path_mlm = folder + dataset_name
    with open(data_path_mlm, 'rb') as file:
        data_mlm = pickle.load(file)
    data_mlm=np.array(data_mlm).tolist()
elif method == "corrot":
    dataset_name = f"corrot/slct{run}.pkl"
    data_path_mlm = folder + dataset_name
    with open(data_path_mlm, 'rb') as file:
        data_mlm = pickle.load(file)
    data_mlm=np.array(data_mlm).tolist()
else:
    print("WRONG METHOD!!!!!")
    
print("Before shuffle: ", data_mlm[:2])
np.random.seed(seed)
# np.random.shuffle(data_mlm)     
    
data_mlm=np.array(data_mlm)[:data_size].tolist()


print("After shuffle: ", data_mlm[:2])
print("Data size: ", len(data_mlm))


tokenizer_mlm = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
inputs_mlm = tokenizer_mlm(data_mlm, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
data_collator_mlm = DataCollatorForLanguageModeling(
    tokenizer=tokenizer_mlm,
    mlm=True,
    mlm_probability=0.15,  # Mask 15% of tokens,
    return_tensors="pt"
)

# Create a training configuration for MLM fine-tuning
training_args_mlm = TrainingArguments(
    output_dir=f"./bert_mlm_finetuned/{model_name}/{method}/{run}/{data_size}/{task_name}",
    overwrite_output_dir=True,
    num_train_epochs=epochs_mlm,
    per_device_train_batch_size=batch_size,
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=learning_rate_mlm,
    weight_decay=0.01,
    fp16=True,
)

# Load the pre-trained BERT model for MLM fine-tuning
model_mlm = AutoModelForMaskedLM.from_pretrained(model_name)


from datasets import Dataset, DatasetDict
dataset = Dataset.from_dict(inputs_mlm)


# Create a Trainer instance for MLM fine-tuning
#This trainer uses Patallel GPU processing
trainer_mlm = Trainer(
    model=model_mlm,
    args=training_args_mlm,
    data_collator=data_collator_mlm,
    train_dataset=dataset,
    tokenizer=tokenizer_mlm,
)

# Fine-tune the model for MLM 
trainer_mlm.train()

to_save = f"./bert_mlm_finetuned/{model_name}/bs{batch_size}/ep_{epochs_mlm}/lr_{learning_rate_mlm}/max{max_length}/seed{seed}/{method}/{task_name}/{data_size}/{run}"

print(to_save)

tokenizer_mlm.save_pretrained(to_save)

trainer_mlm.save_model(to_save)

print("saved")