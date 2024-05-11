from dataclasses import dataclass

TARGET_COLUMN = 'binds'
WHOLE_MOLECULE_COLUMN = 'molecule_smiles'
PROTEIN_COLUMN = 'protein_name'

@dataclass
class ModelTypes:
    XGBOOST = 'xgboost'


@dataclass
class FeaturizerTypes:
    CIRCULAR = 'circular_fingerprint'
    MACCS = 'maccs_fingerprint'