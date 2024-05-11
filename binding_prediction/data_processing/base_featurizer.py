import time
import typing as tp
from abc import ABC

from pyarrow import parquet as pq

from binding_prediction.config.featurizer_config import CircularFingerprintFeaturizerConfig, \
    MACCSFingerprintFeaturizerConfig
from binding_prediction.const import PROTEIN_COLUMN, WHOLE_MOLECULE_COLUMN


class Featurizer(ABC):
    def __init__(self, config: tp.Union[
        CircularFingerprintFeaturizerConfig,
        MACCSFingerprintFeaturizerConfig],
                 pq_file_path: str, protein_map: tp.Dict[str, int], indices: tp.List[int] = None):
        self.config = config
        self.pq_file_path = pq_file_path
        self.protein_map = protein_map
        self.indices = indices

        self.row_group_df = None
        self.smiles = None
        self.proteins_encoded = []

        self.x = None
        self.y = None

    def featurize(self):
        pass

    def process_pq_row_group(self, row_group_number):
        self.prepare_input_smiles(row_group_number)

    def prepare_input_smiles(self, row_group_number):
        print(f"Processing row group {row_group_number}")
        start_time = time.time()
        self.row_group_df = pq.ParquetFile(self.pq_file_path).read_row_group(row_group_number).to_pandas()
        if self.indices is not None:
            self.row_group_df = self.row_group_df.iloc[self.indices]
        print(f"Reading time: {time.time() - start_time}")
        start_time = time.time()
        for protein in self.row_group_df[PROTEIN_COLUMN]:
            if protein not in self.protein_map:
                self.protein_map[protein] = len(self.protein_map)
            self.proteins_encoded.append(self.protein_map[protein])
        print(f"Protein encoding time: {time.time() - start_time}")
        self.smiles = self.row_group_df[WHOLE_MOLECULE_COLUMN]
        self.featurize()
