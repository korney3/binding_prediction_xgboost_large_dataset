import argparse
import os
import time

import numpy as np
import pyarrow.parquet as pq

from binding_prediction.config.config_creation import create_training_config
from binding_prediction.training.training_pipeline import TrainingPipeline
from binding_prediction.utils import calculate_number_of_neg_and_pos_samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_parquet', type=str, default='data/two_row_groups.parquet')
    parser.add_argument('--test_parquet', type=str, default='data/test.parquet')
    parser.add_argument('--config_path', type=str,
                        default='binding_prediction/config/yamls/xgboost_config.yaml')
    parser.add_argument('--debug', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    current_date = time.strftime("%Y-%m-%d_%H-%M-%S")

    parent_logs_dir = os.path.join('logs', current_date)
    os.makedirs(parent_logs_dir, exist_ok=True)

    print('Train validation split')

    train_val_pq = pq.ParquetFile(args.input_parquet)

    neg_samples, pos_samples = calculate_number_of_neg_and_pos_samples(train_val_pq)

    config = create_training_config(train_file_path=args.input_parquet, test_file_path=args.test_parquet,
                                    logs_dir=parent_logs_dir, neg_samples=neg_samples, pos_samples=pos_samples,
                                    config_yaml_path=args.config_path)

    num_ensemble_models = train_val_pq.metadata.num_row_groups // 26

    for i in range(num_ensemble_models):
        pq_group_nums = [i * 26 + j for j in range(26)]

        logs_dir = os.path.join(parent_logs_dir, f'ensemble_model_{i}')
        os.makedirs(logs_dir, exist_ok=True)

        config.training_config.pq_groups_numbers = pq_group_nums
        config.logs_dir = logs_dir

        training_pipeline = TrainingPipeline(config,
                                             debug=args.debug,
                                             rng=np.random.default_rng(seed=42))

        training_pipeline.run()


if __name__ == '__main__':
    main()
