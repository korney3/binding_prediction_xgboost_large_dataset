import json
import os
import time
from typing import Callable, List

import numpy as np
import pyarrow.parquet as pq
import xgboost

from binding_prediction.config.config_creation import Config
from binding_prediction.data_processing.circular_fingerprints import CircularFingerprintFeaturizer
from binding_prediction.const import FeaturizerTypes
from binding_prediction.data_processing.maccs_fingerprint import MACCSFingerprintFeaturizer


class SmilesIterator(xgboost.DataIter):
    def __init__(self, config: Config, file_path: str, indicies: List[int] = None, shuffle: bool = True):
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
            indices_in_shard = np.array(self._shuffled_indices[np.where(
                (self._shuffled_indices >= current_index * self.shard_size) & (
                        self._shuffled_indices < (current_index + 1) * self.shard_size))])
            if len(indices_in_shard) > 0 or self._it == self._num_shards:
                break
            self._it += 1

        if self._it == self._num_shards:
            return 0

        relative_indices = indices_in_shard - current_index * self.shard_size

        if self.config.featurizer_config.name == FeaturizerTypes.CIRCULAR:
            print(f"Number of indicies in shard {len(indices_in_shard)}")
            featurizer = CircularFingerprintFeaturizer(self.config.featurizer_config, self._file_path,
                                                       self._protein_map,
                                                       indices=relative_indices)
        elif self.config.featurizer_config.name == FeaturizerTypes.MACCS:
            featurizer = MACCSFingerprintFeaturizer(self.config.featurizer_config, self._file_path,
                                                    self._protein_map,
                                                    indices=relative_indices)

        else:
            raise NotImplementedError(f"Fingerprint "
                                      f"{self.config.featurizer_config.name} "
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
