from dataclasses import dataclass

import yaml

from binding_prediction.const import ModelTypes


@dataclass
class XGBoostModelConfig:
    name: str
    max_depth: int
    objective: str
    eval_metric: str
    verbosity: int
    nthread: int
    tree_method: str
    grow_policy: str
    subsample: float
    colsample_bytree: float
    num_boost_round: int
    scale_pos_weight: float
    eta: float
    alpha: float


def load_xgboost_model_config_from_yaml(yaml_path: str,
                                        scale_pos_weight=1.0) -> XGBoostModelConfig:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)["model"]
    name = config['name']
    if name not in ModelTypes.__dict__.values():
        raise ValueError(f"Model {name} is not supported")
    config['scale_pos_weight'] = scale_pos_weight
    return XGBoostModelConfig(**config)

