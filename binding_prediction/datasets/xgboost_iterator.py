import json
import os
import time
from typing import Callable, List

import numpy as np
import pyarrow.parquet as pq
import xgboost

from binding_prediction.config.config import Config
from binding_prediction.data_processing.circular_fingerprints import CircularFingerprintFeaturizer
from binding_prediction.const import FeaturizerTypes
from binding_prediction.data_processing.ensemble_predictions_fingerprint import EnsemblePredictionsFeaturizer
from binding_prediction.data_processing.maccs_fingerprint import MACCSFingerprintFeaturizer
from binding_prediction.utils import get_relative_indices


class SmilesIterator(xgboost.DataIter):
    def __init__(self, config: Config, file_path: str,
                 indicies: List[int] = None, shuffle: bool = True):
        self.config = config
        self._file_path = file_path
        self._parquet_filename = os.path.basename(file_path)
        self.parquet_file = pq.ParquetFile(file_path)
        self._dataset_length = self.parquet_file.metadata.num_rows
        self.shard_size = self.parquet_file.metadata.row_group(0).num_rows
        self._shuffle = shuffle
        if indicies is not None:
            self._shuffled_indices = indicies
        else:
            self._shuffled_indices = np.arange(self._dataset_length)
        if shuffle:
            self._shuffled_indices = np.random.permutation(self._shuffled_indices)
        self._num_shards = self.parquet_file.metadata.num_row_groups
        self._protein_map = {}
        self._it = 0
        self._temporary_data = None
        super().__init__(cache_prefix=os.path.join(".", "cache"))

    def next(self, input_data: Callable):
        if self._it == self._num_shards:
            return 0

        with open(self.config.protein_map_path, "r") as f:
            self._protein_map = json.load(f)
        start_time = time.time()
        while True:
            current_index = self._it

            print("Reading row group", current_index)
            relative_indices = get_relative_indices(self._shuffled_indices, current_index, self.shard_size)
            if len(relative_indices) > 0 or self._it == self._num_shards:
                break
            self._it += 1

        if self._it == self._num_shards:
            return 0

        if self.config.yaml_config.featurizer_config.name == FeaturizerTypes.CIRCULAR:
            print(f"Number of indicies in shard {len(relative_indices)}")
            featurizer = CircularFingerprintFeaturizer(self.config, self._file_path,
                                                       self._protein_map,
                                                       indices=relative_indices)
        elif self.config.yaml_config.featurizer_config.name == FeaturizerTypes.MACCS:
            featurizer = MACCSFingerprintFeaturizer(self.config, self._file_path,
                                                    self._protein_map,
                                                    indices=relative_indices)
        elif self.config.yaml_config.featurizer_config.name == FeaturizerTypes.ENSEMBLE_PREDICTIONS:
            featurizer = EnsemblePredictionsFeaturizer(self.config, self._file_path,
                                                       self._protein_map,
                                                       indices=relative_indices)
        else:
            raise NotImplementedError(f"Fingerprint "
                                      f"{self.config.yaml_config.featurizer_config.name} "
                                      f"is not implemented")
        featurizer.process_pq_row_group(current_index)
        x, y = featurizer.x, featurizer.y
        print("Fingerprinting time", time.time() - start_time)
        print("Inputting data")
        start_time = time.time()
        input_data(data=x, label=y)
        print("Inputting time", time.time() - start_time)
        self._it += 1
        return 1

    def reset(self):
        self._it = 0
