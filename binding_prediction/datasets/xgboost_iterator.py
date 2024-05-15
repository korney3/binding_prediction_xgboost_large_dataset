import json
import os
import time
from typing import Callable, List

import numpy as np
import pyarrow.parquet as pq
import xgboost

from binding_prediction.config.config import Config
from binding_prediction.data_processing.base_featurizer import Featurizer
from binding_prediction.data_processing.utils import get_featurizer
from binding_prediction.utils import get_indices_in_shard


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
        while True:
            current_index = self._it

            print("Reading row group", current_index)
            indices_in_shard, relative_indices = get_indices_in_shard(self._shuffled_indices, current_index, self.shard_size)
            if len(relative_indices) > 0 or self._it == self._num_shards:
                break
            self._it += 1

        if self._it == self._num_shards:
            return 0
        print(f"Number of indices in shard {len(relative_indices)}")
        featurizer = get_featurizer(self.config, self._file_path)
        featurizer.process_pq_row_group(current_index, indices_in_shard, relative_indices)
        x, y = featurizer.x, featurizer.y
        input_data(data=x, label=y)
        self._it += 1
        return 1

    def reset(self):
        self._it = 0
