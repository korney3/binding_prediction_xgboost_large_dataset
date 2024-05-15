import json
import time
import typing as tp
from abc import ABC
from multiprocessing import Pool

import numpy as np
from pyarrow import parquet as pq

from binding_prediction.config.config import Config
from binding_prediction.const import PROTEIN_COLUMN, WHOLE_MOLECULE_COLUMN
from binding_prediction.const import TARGET_COLUMN


class Featurizer(ABC):
    def __init__(self, config: Config, pq_file_path: str):

        self.featurizer_config = config.yaml_config.featurizer_config
        self.pq_file_path = pq_file_path

        with open(config.protein_map_path, "r") as f:
            self.protein_map = json.load(f)

        self.row_group_df = None
        self.smiles = None
        self.proteins_encoded = []

        self.relative_indices = None
        self.indices_in_shard = None

        self.x = None
        self.y = None

    def featurize(self):
        pass

    def process_pq_row_group(self, row_group_number,
                             indices_in_shard: tp.List[int],
                             relative_indices: tp.List[int]):
        self.relative_indices = relative_indices
        self.indices_in_shard = indices_in_shard
        self.prepare_input_smiles(row_group_number)

    def prepare_input_smiles(self, row_group_number):
        print(f"Processing row group {row_group_number}")
        self.row_group_df = pq.ParquetFile(self.pq_file_path).read_row_group(row_group_number).to_pandas()
        self.row_group_df = self.row_group_df.iloc[self.relative_indices]
        for protein in self.row_group_df[PROTEIN_COLUMN]:
            if protein not in self.protein_map:
                self.protein_map[protein] = len(self.protein_map)
            self.proteins_encoded.append(self.protein_map[protein])
        self.smiles = self.row_group_df[WHOLE_MOLECULE_COLUMN]
        self.featurize()

    def _featurize(self, smiles_to_fingerprint):
        start_time = time.time()
        with Pool(8) as p:
            self.x = np.array(p.map(smiles_to_fingerprint, self.smiles))
        print(f"Fingerprinting time: {time.time() - start_time}")
        self.add_protein_encoded_feature()
        self.create_target()

    def add_protein_encoded_feature(self):
        self.x = np.array([self.x[i] + [self.proteins_encoded[i]] for i in range(len(self.x))])

    def create_target(self):
        if TARGET_COLUMN in self.row_group_df.columns:
            self.y = np.array(self.row_group_df[TARGET_COLUMN])
        else:
            self.y = np.array([-1] * len(self.x))
