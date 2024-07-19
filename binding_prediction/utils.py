import logging
import os
import time
import typing as tp
from logging import Logger

import numpy as np
import yaml
from pyarrow import parquet as pq

from binding_prediction.config.config import Config, create_config
from binding_prediction.const import ModelTypes, WEAK_LEARNER_ARTIFACTS_NAME_PREFIX, DEFAULT_NUM_WEAK_LEARNERS


def get_indices_in_shard(indices, current_shard_num, shard_size):
    indices_in_shard = np.array(indices[np.where(
        (indices >= current_shard_num * shard_size) & (
                indices < (current_shard_num + 1) * shard_size))])
    relative_indices = indices_in_shard - current_shard_num * shard_size
    return indices_in_shard, relative_indices


def calculate_number_of_neg_and_pos_samples(pq_file,
                                            indices=None):
    neg_samples = 0
    pos_samples = 0
    shard_size = pq_file.metadata.row_group(0).num_rows
    for group_id in range(pq_file.metadata.num_row_groups):
        group_df = pq_file.read_row_group(group_id).to_pandas()
        if indices is not None:
            _, relative_indices = get_indices_in_shard(indices, group_id, shard_size)
            group_df = group_df.iloc[relative_indices]

        neg_samples += len(group_df[group_df['binds'] == 0])
        pos_samples += len(group_df[group_df['binds'] == 1])
    return neg_samples, pos_samples


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return result

    return wrapper


def pretty_print_text(text):
    pretty_text = f"{'=' * len(text)}\n{text.upper()}\n{'=' * len(text)}"
    print(pretty_text)
    return pretty_text


def save_config(config):
    with open(os.path.join(config.logs_dir, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)


def load_config(logs_dir) -> Config:
    with open(os.path.join(logs_dir, "config.yaml"), "r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    return config


def create_logs_dir(logs_dir_location: str = 'logs') -> str:
    current_date = time.strftime("%Y-%m-%d_%H-%M-%S")
    logs_dir = os.path.join(logs_dir_location, current_date)
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def get_config(train_parquet_path: str, test_parquet_path: str,
               config_obj: tp.Union[str, dict], logs_dir: str,
               train_val_indices: tp.Optional[list] = None) -> Config:

    train_val_pq = pq.ParquetFile(train_parquet_path)
    neg_samples, pos_samples = calculate_number_of_neg_and_pos_samples(train_val_pq, indices=train_val_indices)

    config = create_config(train_file_path=train_parquet_path, test_file_path=test_parquet_path,
                           logs_dir=logs_dir, neg_samples=neg_samples, pos_samples=pos_samples,
                           config=config_obj)
    if config.yaml_config.model_config.name == ModelTypes.XGBOOST_ENSEMBLE:
        weak_learners_train_size = train_val_pq.metadata.num_rows
        if config.yaml_config.training_config.train_size != -1:
            weak_learners_train_size = train_val_pq.metadata.num_rows - config.yaml_config.training_config.train_size
            if weak_learners_train_size <= 0:
                raise ValueError(f"Train size of ensemble model is too large, "
                                 f"cannot split data to train multiple weak learners and ensemble model. "
                                 f"Decrease train_size of ensemble model")
        num_weak_learners = (weak_learners_train_size //
                             config.yaml_config.model_config.weak_learner_config["train"]["train_size"])
        if num_weak_learners == 0:
            raise ValueError(f"Train size of weak learner is too large, cannot create multiple weak learners"
                             f"Decrease train_size of weak learner")

        config.yaml_config.model_config.num_weak_learners = num_weak_learners
    return config


def save_weak_learners_data_indices(train_parquet_path, ensemble_config, parent_logs_dir, rng):
    train_val_pq = pq.ParquetFile(train_parquet_path)
    num_weak_learners = ensemble_config.yaml_config.model_config.num_weak_learners
    weak_learners_train_size = train_val_pq.metadata.num_rows
    if ensemble_config.yaml_config.training_config.train_size != -1:
        weak_learners_train_size = train_val_pq.metadata.num_rows - ensemble_config.yaml_config.training_config.train_size
    all_train_val_indices = np.arange(weak_learners_train_size)
    all_weak_learner_indices = rng.choice(all_train_val_indices, size=(
        num_weak_learners, ensemble_config.yaml_config.model_config.weak_learner_config["train"]["train_size"]),
                                          replace=False)
    all_indices = np.arange(train_val_pq.metadata.num_rows)
    final_ensemble_model_indices = np.setdiff1d(all_indices, all_weak_learner_indices.flatten())
    for i in range(num_weak_learners):
        np.save(os.path.join(parent_logs_dir, f'{WEAK_LEARNER_ARTIFACTS_NAME_PREFIX}{i}_indices.npy'),
                all_weak_learner_indices[i])
    np.save(os.path.join(parent_logs_dir, 'final_ensemble_model_indices.npy'), final_ensemble_model_indices)
