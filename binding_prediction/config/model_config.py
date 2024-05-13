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
    device: str = 'cpu'


def load_xgboost_model_config_from_yaml_path(yaml_path: str,
                                             scale_pos_weight=1.0) -> XGBoostModelConfig:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return create_xgboost_model_config_from_dict(config, scale_pos_weight)


def create_xgboost_model_config_from_dict(config: dict,
                                          scale_pos_weight=1.0) -> XGBoostModelConfig:
    model_config_dict = config["model"]
    name = model_config_dict['name']
    if name not in ModelTypes.__dict__.values():
        raise ValueError(f"Model {name} is not supported")
    if 'scale_pos_weight' not in model_config_dict:
        model_config_dict['scale_pos_weight'] = scale_pos_weight
    return XGBoostModelConfig(**model_config_dict)


@dataclass
class XGBoostEnsembleModelConfig(XGBoostModelConfig):
    weak_learner_config: dict = None
    num_weak_learners: int = -1


def load_xgboost_ensemble_model_config_from_yaml_path(yaml_path: str,
                                                      scale_pos_weight=1.0) -> XGBoostEnsembleModelConfig:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return create_xgboost_ensemble_model_config_from_dict(config, scale_pos_weight)


def create_xgboost_ensemble_model_config_from_dict(config: dict,
                                                   scale_pos_weight=1.0) -> XGBoostEnsembleModelConfig:
    model_config_dict = config["model"]
    name = model_config_dict['name']
    if name not in ModelTypes.__dict__.values():
        raise ValueError(f"Model {name} is not supported")
    if 'scale_pos_weight' not in model_config_dict:
        model_config_dict['scale_pos_weight'] = scale_pos_weight
    return XGBoostEnsembleModelConfig(**model_config_dict)
