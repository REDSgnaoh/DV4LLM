import json
import logging
import math
from functools import partial
from pathlib import Path
from typing import Iterable, List

import openai
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers.pipelines import pipeline

from gpt2_generation import GPT2Generation
OPENAI_API_KEY = ""

logging.disable(logging.CRITICAL)  # Disable logging from transformers

def load_cache(file: Path):
    if file.exists():
        with file.open() as f:
            for line in tqdm(f, desc=f'Loading cache from {file}'):
                yield json.loads(line)

def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch

def _gpt2_helper(prompts: pd.Series,
                 max_len: int,
                 num_samples: int,
                 batch_size: int,
                 generator: GPT2Generation,
                 out_file: Path = None,
                 **generate_kwargs):
    # Repeat prompts
    prompts = prompts.repeat(num_samples)

    # Resume generation
    num_cached_generations = 0
    for generation in load_cache(out_file):
        yield generation
        num_cached_generations += 1

    # Generate with prompts
    prompts = prompts[num_cached_generations:]
    for prompt in tqdm(batchify(prompts, batch_size),
                       total=math.ceil(len(prompts) / batch_size),
                       desc=f'GPT-2 Generation',
                       dynamic_ncols=True,
                       postfix={'batch_size': batch_size},
                       disable=True):
        # Generate
        try:
            batch = generator.generate(prompt, max_len, **generate_kwargs)
        except RuntimeError as e:
            print("Error during generation with prompt:", prompt)
            print(e)
            print("Emptying CUDA cache and retrying...")
            torch.cuda.empty_cache()

            batch = ["GENERATION_ERROR_CUDA"] * len(prompt)

        for generation in batch:
            with out_file.open('a') as f:
                print(json.dumps(generation), file=f)
            yield generation


def gpt2(prompts: pd.Series,
         max_len: int,
         num_samples: int,
         batch_size: int,
         model_name_or_path: str,
         seed: int = 42,
         out_file: Path = None,
         **generate_kwargs) -> Iterable[str]:
    # Setup model
    generator = GPT2Generation(model_name_or_path, seed=seed)

    yield from _gpt2_helper(prompts=prompts,
                            max_len=max_len,
                            num_samples=num_samples,
                            batch_size=batch_size,
                            generator=generator,
                            out_file=out_file,
                            **generate_kwargs)


def gpt3(prompts: pd.Series,
         max_len: int,
         num_samples: int,
         batch_size: int,
         model_name_or_path: str,
         out_file: Path) -> Iterable[str]:
    openai.api_key = OPENAI_API_KEY

    def request(prompts: List[str]):
        # Retry request (handles connection errors, timeouts, and overloaded API)
        while True:
            try:
                return openai.Completion.create(
                    engine=model_name_or_path,
                    prompt=prompts,
                    max_tokens=max_len,
                    n=1
                )
            except Exception as e:
                tqdm.write(str(e))
                tqdm.write("Retrying...")

    prompts = prompts.repeat(num_samples)
    for batch in tqdm(batchify(prompts, batch_size)):
        response = request(batch)
        yield from [choice['text'] for choice in response['choices']]
