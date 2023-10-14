from typing import List, Iterator, Optional
import argparse
import sys
import torch
import json

from vampire_reader import VampireReader
from vampire import VAMPIRE
from allennlp_bridge import ExtendedVocabulary


from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import lazy_groups_of, sanitize
from allennlp.models.archival import load_archive
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance
from tqdm import tqdm

import numpy as np


def main():
    # Specify the parameters directly
    archive_file = "train_logs/arxiv/model.tar.gz"
    # for train domain
    input_file = "train_data/arxiv.jsonl"
    # for glue
    # "Target_datasets/cs_scierc_id_train.jsonl"
    # input_file = "Target_datasets/cs_scierc_id_train.jsonl"
    # "Embeddings/cs_sci/embedding.npz"
    output_file = "Embeddings/arxiv/embedding.npz"
    weights_file = ""
    batch_size = 128
    silent = False
    cuda_device = 6
    use_dataset_reader = False
    dataset_reader_choice = "train"
    overrides = ""
    predictor = "Vampire_Predictor"

    # Call the existing functions with the specified parameters
    Predictor = _get_predictor(
        archive_file=archive_file,
        weights_file=weights_file,
        cuda_device=cuda_device,
        overrides=overrides,
        dataset_reader_choice=dataset_reader_choice,
        predictor=predictor,
    )

    # Initialize and run the predict manager
    manager = _PredictManager(
        predictor=Predictor,
        input_file=input_file,
        output_file=output_file,
        batch_size=batch_size,
        print_to_console=not silent,
        has_dataset_reader=use_dataset_reader,
    )
    manager.run()

def _get_predictor(archive_file, weights_file, cuda_device, overrides, predictor, dataset_reader_choice) -> Predictor:
    check_for_gpu(cuda_device)
    print("loading from archive...")
    archive = load_archive(
        archive_file,
        # weights_file=weights_file,
        cuda_device=cuda_device,
        overrides=overrides,
    )
    print("done!")
   
    return Predictor.from_archive(
        archive, predictor, dataset_reader_to_load=dataset_reader_choice
    )

class _PredictManager:
    def __init__(
        self,
        predictor: Predictor,
        input_file: str,
        output_file: Optional[str],
        batch_size: int,
        print_to_console: bool,
        has_dataset_reader: bool,
    ) -> None:

        self._predictor = predictor
        self._input_file = input_file
        if output_file is not None:
            self._output_file = output_file
        else:
            self._output_file = None
        self._batch_size = batch_size
        self._print_to_console = print_to_console
        if has_dataset_reader:
            self._dataset_reader = predictor._dataset_reader
        else:
            self._dataset_reader = None
        num_layers = len(self._predictor._model.vae.encoder._linear_layers) + 1  # pylint: disable=protected-access
        initial_params = [1] + [-20] * (num_layers - 2) + [1]
        self.scalar_mix = ScalarMix(
                num_layers,
                do_layer_norm=False,
                initial_scalar_parameters=initial_params,
                trainable=False)

    # required
    def _predict_json(self, batch_data: List[JsonDict]) -> Iterator[str]:
        if len(batch_data) == 1:
            results = [self._predictor.predict_json(batch_data[0])]
        else:
            results = self._predictor.predict_batch_json(batch_data)
        for output in results:
            yield output


    def _predict_instances(self, batch_data: List[Instance]) -> Iterator[str]:
        if len(batch_data) == 1:
            results = [self._predictor.predict_instance(batch_data[0])]
        else:
            results = self._predictor.predict_batch_instance(batch_data)
        for output in results:
            yield output

    def _maybe_print_to_console_and_file(
        self, index: int, prediction: str, model_input: str = None
    ) -> None:
        if self._print_to_console:
            if model_input is not None:
                print(f"input {index}: ", model_input)
            print("prediction: ", prediction)
        if self._output_file is not None:
            self._output_file.write(prediction)

    # required
    def _get_json_data(self) -> Iterator[JsonDict]:
        if self._input_file == "-":
            for line in sys.stdin:
                if not line.isspace():
                    yield self._predictor.load_line(line)
        else:
            input_file = cached_path(self._input_file)
            with open(input_file, "r") as file_input:
                for line in file_input:
                    if not line.isspace():
                        yield self._predictor.load_line(line)
                        

    def _get_instance_data(self) -> Iterator[Instance]:
        if self._input_file == "-":
            raise ConfigurationError("stdin is not an option when using a DatasetReader.")
        elif self._dataset_reader is None:
            raise ConfigurationError("To generate instances directly, pass a DatasetReader.")
        else:
            yield from self._dataset_reader.read(self._input_file)

    def run(self) -> None:
        
        has_reader = self._dataset_reader is not None
        index = 0

        if has_reader:
            for batch in lazy_groups_of(self._get_instance_data(), self._batch_size):
                for model_input_instance, result in zip(batch, self._predict_instances(batch)):
                    self._maybe_print_to_console_and_file(index, result, str(model_input_instance))
                    index = index + 1
        else:
            ids_ = []
            vecs = []
            for batch_json in tqdm(lazy_groups_of(self._get_json_data(), self._batch_size)):
                for model_input_json, result in zip(batch_json, self._predict_json(batch_json)):
                    # print("result_keys: ", result.keys())
                    # print("model_input_json_keys: ", model_input_json.keys())
                    # print("model_input_json_ids: ", model_input_json['id'])

                    # scalar_mix = (torch.Tensor(result['activation_encoder_layer_0']).unsqueeze(0)
                    #                 + -20 * torch.Tensor(result['activation_encoder_layer_1']).unsqueeze(0)
                    #                 + torch.Tensor(result['activation_theta']).unsqueeze(0))

                    scalar_mix = (torch.Tensor(result['activations'][0][1]).unsqueeze(0)
                                    + -20 * torch.Tensor(result['activations'][1][1]).unsqueeze(0)
                                    + torch.Tensor(result['theta']).unsqueeze(0))
                
                    reshaped_scalar_mix = scalar_mix.reshape(1, scalar_mix.shape[2])

                    vecs.append(reshaped_scalar_mix)
                    # for tensor id
                    # ids_.append(torch.Tensor([model_input_json['id']]).unsqueeze(0))
                    # for str id
                    ids_.append(model_input_json['id'])
                    index = index + 1
            # for tensor id
            # torch.save((torch.cat(ids_,0), torch.cat(vecs, 0)), self._output_file)
            # for str id
            numpy_vecs = torch.cat(vecs, 0).numpy()
            ids_ = np.array(ids_, dtype=object)
            np.savez(self._output_file, ids_=ids_, vecs_=numpy_vecs)

# Run the main function
if __name__ == '__main__':
    main()