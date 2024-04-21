import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import matplotlib.pyplot as plt
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm
import math
import xgboost
import xgboost as xgb
import os
from typing import Callable, List
import time

from binding_prediction.const import WHOLE_MOLECULE_COLUMN, PROTEIN_COLUMN, TARGET_COLUMN
from binding_prediction.data_processing.circular_fingerprints import smiles_to_fingerprint


class SmilesIterator(xgboost.DataIter):
    def __init__(self, file_path: str,
                 indicies: List[int] = None,
                 shuffle: bool = True,
                 test_set: bool = True,
                 fingerprint="circular",
                 radius=2,
                 nBits=2048):
        self._file_path = file_path
        self._parquet_file = pq.ParquetFile(file_path)
        self._dataset_length = self._parquet_file.metadata.num_rows
        self._num_shards = self._parquet_file.metadata.num_row_groups
        self._shard_size = self._dataset_length // self._num_shards
        self._shuffle = shuffle
        if indicies is not None:
            self._shuffled_indices = indicies
        else:
            self._shuffled_indices = np.arange(self._dataset_length)
        if shuffle:
            self._shuffled_indices = np.random.permutation(self._shuffled_indices)

        self._protein_map = {}
        if test_set:
            self._cache_path = os.path.join("data/processed", f"{fingerprint}_{radius}_{nBits}/test")
        else:
            self._cache_path = os.path.join("data/processed", f"{fingerprint}_{radius}_{nBits}/train")
        self._it = 0
        self._temporary_data = None
        super().__init__(cache_prefix=os.path.join(".", "cache"))

    def next(self, input_data: Callable):
        if self._it == self._num_shards:
            return 0
        print("Reading row group", self._it)
        start_time = time.time()

        indicies_in_shard = np.array(self._shuffled_indices[np.where(
            (self._shuffled_indices >= self._it * self._shard_size) & (
                    self._shuffled_indices < (self._it + 1) * self._shard_size))])
        relative_indicies = indicies_in_shard - self._it * self._shard_size

        if os.path.exists(os.path.join(self._cache_path, f"Commit_file_{self._it}.txt")):
            start_time = time.time()
            x_all = np.load(os.path.join(self._cache_path, f"x_{self._it}.npy"))
            y_all = np.load(os.path.join(self._cache_path, f"y_{self._it}.npy"))
            input_data(data=x_all[relative_indicies], label=y_all[relative_indicies])
            print("Reading time", time.time() - start_time)
            self._it += 1
            return 1

        row_group_df = self._parquet_file.read_row_group(self._it).to_pandas()

        row_group_df = row_group_df.iloc[relative_indicies]
        print("Reading time", time.time() - start_time)
        print("Row group shape", row_group_df.shape)
        start_time = time.time()
        smiles = row_group_df[WHOLE_MOLECULE_COLUMN]
        proteins = row_group_df[PROTEIN_COLUMN]
        if TARGET_COLUMN not in row_group_df.columns:
            target = [0] * len(row_group_df)
        else:
            target = row_group_df[TARGET_COLUMN]
        self._it += 1

        encoded_protein = []
        for protein in proteins:
            if protein not in self._protein_map:
                self._protein_map[protein] = len(self._protein_map)
            encoded_protein.append(self._protein_map[protein])
        print("Encoding time", time.time() - start_time)
        print("Converting to fingerprints")
        start_time = time.time()
        x = np.array([smiles_to_fingerprint(s) + [encoded_protein[i]] for i, s in tqdm(enumerate(smiles))])
        target = np.array(target)
        print("Fingerprinting time", time.time() - start_time)
        print("Inputting data")
        start_time = time.time()
        input_data(data=x, label=target)
        print("Inputting time", time.time() - start_time)
        return 1

    def reset(self):
        self._it = 0
