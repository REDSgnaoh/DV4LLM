import json
import os
import random
import shutil
import subprocess
import tempfile

from allennlp.common.params import Params
from environment import ENVIRONMENTS
from random_search import HyperparameterSearch

random_int = random.randint(0, 2**32)

def main():
    # Define your arguments here
    override = True  # or False
    config = "config/vampire.jsonnet"  # replace with your config path
    serialization_dir = "train_logs/arxiv"  # replace with your serialization directory
    environment = "VAMPIRE"  # replace with your environment
    recover = False # or False
    device = "4"  # replace with your device, or None if not using
    seed = ""  # replace with your seed, or None if not using

    env = ENVIRONMENTS[environment.upper()]

    space = HyperparameterSearch(**env)
    
    sample = space.sample()

    for key, val in sample.items():
        os.environ[key] = str(val)

    if device:
        os.environ['CUDA_DEVICE'] = device

    if seed:
        os.environ['SEED'] = seed

    allennlp_command = [
            "allennlp",
            "train",
            config,
            "--include-package",
            "vampire",
            "--include-package",
            "vampire_reader",
            "--include-package",
            "allennlp_bridge",
            "-s",
            serialization_dir
            ]

    if seed:
        allennlp_command[-1] = allennlp_command[-1] + "_" + seed

    if recover:
        def append_seed_to_config(seed, serialization_dir):
            seed = str(seed)
            seed_dict = {"pytorch_seed": seed,
                         "random_seed": seed,
                         "numpy_seed": seed}
            config_path = os.path.join(serialization_dir, 'config.json')
            with open(config_path, 'r+') as f:
                config_dict = json.load(f)
                seed_dict.update(config_dict)
                f.seek(0)
                json.dump(seed_dict, f, indent=4)

        append_seed_to_config(seed=seed, serialization_dir=allennlp_command[-1])

        allennlp_command.append("--recover")

    if os.path.exists(allennlp_command[-1]) and override:
        print(f"overriding {allennlp_command[-1]}")
        shutil.rmtree(allennlp_command[-1])

    subprocess.run(" ".join(allennlp_command), shell=True, check=True)


if __name__ == '__main__':
    main()
