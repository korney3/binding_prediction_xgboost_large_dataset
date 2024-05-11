import time
import typing as tp
from functools import partial
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from binding_prediction.config.featurizer_config import CircularFingerprintFeaturizerConfig
from binding_prediction.const import TARGET_COLUMN
from binding_prediction.data_processing.base_featurizer import Featurizer


class CircularFingerprintFeaturizer(Featurizer):
    def __init__(self, config: CircularFingerprintFeaturizerConfig,
                 pq_file_path: str, protein_map: tp.Dict[str, int],
                 indices: tp.List[int] = None):
        super().__init__(config, pq_file_path, protein_map, indices)
        self.config = config

    def featurize(self):
        start_time = time.time()
        partial_smiles_to_fingerprint = (
            partial(smiles_to_fingerprint, nBits=self.config.length,
                    radius=self.config.radius))
        with Pool(8) as p:
            x = np.array(p.map(partial_smiles_to_fingerprint, self.smiles))
        print(f"Fingerprinting time: {time.time() - start_time}")
        start_time = time.time()
        self.x = np.array([x[i] + [self.proteins_encoded[i]] for i in range(len(x))])
        if TARGET_COLUMN in self.row_group_df.columns:
            self.y = np.array(self.row_group_df[TARGET_COLUMN])
        else:
            self.y = np.array([-1] * len(x))
        print(f"Combining time: {time.time() - start_time}")


def smiles_to_fingerprint(smiles, nBits=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
