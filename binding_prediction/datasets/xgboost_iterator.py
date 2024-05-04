import os
import time
from typing import Callable, List

import numpy as np
import pyarrow.parquet as pq
import xgboost

from binding_prediction.data_processing.circular_fingerprints import create_circular_fingerprints_from_pq_row_group


class SmilesIterator(xgboost.DataIter):
    def __init__(self, file_path: str,
                 indicies: List[int] = None,
                 shuffle: bool = True,
                 fingerprint="circular",
                 radius=2,
                 nBits=2048,
                 protein_map_path=None):
        self._file_path = file_path
        self._parquet_filename = os.path.basename(file_path)
        self.parquet_file = pq.ParquetFile(file_path)
        self._dataset_length = self.parquet_file.metadata.num_rows
        self._num_shards = self.parquet_file.metadata.num_row_groups
        self.shard_size = self.parquet_file.metadata.row_group(0).num_rows
        self._shuffle = shuffle
        self._radius = radius
        self._fingerprint_length = nBits
        if indicies is not None:
            self._shuffled_indices = indicies
        else:
            self._shuffled_indices = np.arange(self._dataset_length)
        if shuffle:
            self._shuffled_indices = np.random.permutation(self._shuffled_indices)
        self._cache_path = os.path.join("data/processed", self._parquet_filename,
                                        f"{fingerprint}_{self._radius}_{self._fingerprint_length}")
        os.makedirs(self._cache_path, exist_ok=True)
        if protein_map_path is not None:
            self.protein_map_path = protein_map_path
        else:
            self.protein_map_path = os.path.join(self._cache_path, "protein_map.npy")
        self._protein_map = {}
        self._it = 0
        self._temporary_data = None
        super().__init__(cache_prefix=os.path.join(".", "cache"))

    def next(self, input_data: Callable):
        if self._it == self._num_shards:
            with open(self.protein_map_path, "wb") as f:
                np.save(f, self._protein_map)
            return 0
        print("Reading row group", self._it)
        if os.path.exists(self.protein_map_path):
            self._protein_map = np.load(self.protein_map_path, allow_pickle=True).item()
        start_time = time.time()

        indicies_in_shard = np.array(self._shuffled_indices[np.where(
            (self._shuffled_indices >= self._it * self.shard_size) & (
                    self._shuffled_indices < (self._it + 1) * self.shard_size))])
        relative_indicies = indicies_in_shard - self._it * self.shard_size

        if os.path.exists(os.path.join(self._cache_path, f"Commit_file_{self._it}.txt")):
            start_time = time.time()
            x_all = np.load(os.path.join(self._cache_path, f"x_{self._it}.npy"))
            y_all = np.load(os.path.join(self._cache_path, f"y_{self._it}.npy"))
            input_data(data=x_all[relative_indicies], label=y_all[relative_indicies])
            print("Reading time", time.time() - start_time)
            self._it += 1
            return 1
        input_smiles, x, y = create_circular_fingerprints_from_pq_row_group(self._file_path, self._it,
                                                                            self._protein_map,
                                                                            self._radius,
                                                                            self._fingerprint_length)
        x = x[relative_indicies]
        y = y[relative_indicies]
        print("Fingerprinting time", time.time() - start_time)
        print("Inputting data")
        start_time = time.time()
        input_data(data=x, label=y)
        print("Inputting time", time.time() - start_time)
        self._it += 1
        return 1

    def reset(self):
        self._it = 0
