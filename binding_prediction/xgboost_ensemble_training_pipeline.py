import argparse
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from binding_prediction.config.config import create_config
from binding_prediction.training.training_pipeline import TrainingPipeline
from binding_prediction.utils import calculate_number_of_neg_and_pos_samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_parquet', type=str, default='data/two_row_groups.parquet')
    parser.add_argument('--test_parquet', type=str, default='data/test.parquet')
    parser.add_argument('--config_path', type=str,
                        default='binding_prediction/config/yamls/xgboost_ensemble_config.yaml')
    parser.add_argument('--debug', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    current_date = time.strftime("%Y-%m-%d_%H-%M-%S")

    parent_logs_dir = os.path.join('logs', current_date)
    parent_logs_dir = "logs/2024-05-12_00-36-58"
    os.makedirs(parent_logs_dir, exist_ok=True)

    rng = np.random.default_rng(seed=42)

    train_val_pq = pq.ParquetFile(args.input_parquet)

    with open(args.config_path, 'r') as file:
        train_config_dict = yaml.safe_load(file)["train"]
    if "pq_groups_numbers" in train_config_dict and train_config_dict["pq_groups_numbers"] is not None:
        pq_groups_numbers = sorted(train_config_dict["pq_groups_numbers"])
        pq_groups_numbers = list(filter(lambda x: x < train_val_pq.num_row_groups, pq_groups_numbers))
    else:
        pq_groups_numbers = None

    neg_samples, pos_samples = calculate_number_of_neg_and_pos_samples(train_val_pq, pq_groups_numbers)

    ensemble_config = create_config(train_file_path=args.input_parquet, test_file_path=args.test_parquet,
                                    logs_dir=parent_logs_dir, neg_samples=neg_samples, pos_samples=pos_samples,
                                    config=args.config_path)

    num_weak_learners = (train_val_pq.metadata.num_rows //
                         ensemble_config.yaml_config.model_config.weak_learner_config["train"]["train_size"])
    ensemble_config.yaml_config.model_config.num_weak_learners = num_weak_learners
    # save_weak_learners_data_indices(train_val_pq, ensemble_config, num_weak_learners,
    #                                 parent_logs_dir, rng)
    #
    # for i in range(num_weak_learners):
    #     train_weak_learner(train_val_pq, args, i, parent_logs_dir)
    final_ensemble_model_indices = np.load(os.path.join(parent_logs_dir, 'final_ensemble_model_indices.npy'))
    training_pipeline = TrainingPipeline(ensemble_config, debug=args.debug, rng=rng,
                                         train_val_indices=final_ensemble_model_indices)
    training_pipeline.run()


def train_weak_learner(train_val_pq, args, weak_learner_num, parent_logs_dir):
    train_val_indices = np.load(os.path.join(parent_logs_dir, f'weak_learner_{weak_learner_num}_indices.npy'))
    logs_dir = os.path.join(parent_logs_dir, f'weak_learner_{weak_learner_num}')
    os.makedirs(logs_dir, exist_ok=True)
    neg_samples, pos_samples = calculate_number_of_neg_and_pos_samples(train_val_pq,
                                                                       indices=train_val_indices)
    with open(args.config_path, 'r') as file:
        weak_learner_config_dict = yaml.safe_load(file)["model"]["weak_learner_config"]
    config = create_config(train_file_path=args.input_parquet, test_file_path=args.test_parquet,
                           logs_dir=logs_dir, neg_samples=neg_samples, pos_samples=pos_samples,
                           config=weak_learner_config_dict)
    training_pipeline = TrainingPipeline(config,
                                         debug=args.debug,
                                         rng=np.random.default_rng(seed=42),
                                         train_val_indices=train_val_indices)
    training_pipeline.run()
    save_predictions_for_ensemble_model(training_pipeline, weak_learner_num, parent_logs_dir)
    delete_xgboost_cache()


def save_predictions_for_ensemble_model(training_pipeline, weak_learner_num, parent_logs_dir):
    final_ensemble_model_indices = np.load(os.path.join(parent_logs_dir, 'final_ensemble_model_indices.npy'))
    final_ensemble_model_predictions = training_pipeline.predict(final_ensemble_model_indices)
    final_ensemble_model_predictions_df = pd.DataFrame(columns=['index_in_train_file', 'prediction'])
    final_ensemble_model_predictions_df['index_in_train_file'] = final_ensemble_model_indices
    final_ensemble_model_predictions_df['prediction'] = final_ensemble_model_predictions
    final_ensemble_model_predictions_df.to_csv(
        os.path.join(parent_logs_dir, f'final_ensemble_model_predictions_{weak_learner_num}.csv'),
        index=False)


def delete_xgboost_cache():
    cache_files = list(filter(lambda x: x.startswith("cache-0") and x.endswith(".page"), os.listdir(".")))
    cache_files_paths = list(map(lambda x: os.path.join(".", x), cache_files))
    for cache_file in cache_files_paths:
        os.remove(cache_file)


def save_weak_learners_data_indices(train_val_pq, ensemble_config, num_weak_learners, parent_logs_dir, rng):
    all_train_val_indices = np.arange(train_val_pq.metadata.num_rows)
    all_weak_learner_indices = rng.choice(all_train_val_indices, size=(
        num_weak_learners, ensemble_config.yaml_config.model_config.weak_learner_config["train"]["train_size"]),
                                          replace=False)
    final_ensemble_model_indices = np.setdiff1d(all_train_val_indices, all_weak_learner_indices.flatten())
    for i in range(num_weak_learners):
        np.save(os.path.join(parent_logs_dir, f'weak_learner_{i}_indices.npy'), all_weak_learner_indices[i])
    np.save(os.path.join(parent_logs_dir, 'final_ensemble_model_indices.npy'), final_ensemble_model_indices)


if __name__ == '__main__':
    main()
