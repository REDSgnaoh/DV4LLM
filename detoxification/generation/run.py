import pickle
import pandas as pd
from generation import gpt2
import json
import tqdm
import argparse

def generate_and_save_text(prompts, indices):
    generated_texts = []
    prompt_series = pd.Series(prompts)
    for generated_batch in gpt2(
            prompts=prompt_series,
            max_len=max_len,
            num_samples=num_samples,
            batch_size=batch_size,
            model_name_or_path=model_name_or_path):
        generated_texts.append(generated_batch)
    for i, index in enumerate(indices):
        output_dict[index] = generated_texts[i * num_samples: (i + 1) * num_samples]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save text.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--output_file_path", type=str, required=True, help="Path to the output file.")
    args = parser.parse_args()

    with open(args.data_file, "rb") as f:
        toxic_data = pickle.load(f)

    output_dict = {}
    max_len = 20
    num_samples = 25
    batch_size = 500
    model_name_or_path = args.model_name_or_path
    output_file_path = args.output_file_path

    prompt_batches = [toxic_data[i:i+batch_size//num_samples] for i in range(0, len(toxic_data), batch_size//num_samples)]
    for batch_index in tqdm.tqdm(range(len(prompt_batches))):
        prompt_batch = prompt_batches[batch_index]
        generate_and_save_text(prompt_batch, range(batch_index * batch_size//num_samples, (batch_index + 1) * batch_size//num_samples))
        with open(output_file_path, "w") as output_file:
            json.dump(output_dict, output_file)
