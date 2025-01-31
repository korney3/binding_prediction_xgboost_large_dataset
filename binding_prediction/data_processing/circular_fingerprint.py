import typing as tp
from functools import partial
from logging import Logger

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from binding_prediction.config.config import Config
from binding_prediction.data_processing.base_featurizer import Featurizer


class CircularFingerprintFeaturizer(Featurizer):
    def __init__(self, config: Config,
                 pq_file_path: str, logger: tp.Optional[Logger] = None):
        super().__init__(config, pq_file_path, logger=logger)

    def featurize(self):
        partial_smiles_to_fingerprint = (
            partial(smiles_to_fingerprint, nBits=self.featurizer_config.length,
                    radius=self.featurizer_config.radius))
        self._featurize(partial_smiles_to_fingerprint)


def smiles_to_fingerprint(smiles, nBits=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
