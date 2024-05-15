import argparse
import os
import time

import numpy as np
import pyarrow.parquet as pq
import yaml

from binding_prediction.config.config import create_config
from binding_prediction.evaluation.utils import evaluate_test_set, evaluate_validation_set
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

    logs_dir = os.path.join('logs', current_date)
    os.makedirs(logs_dir, exist_ok=True)

    train_val_pq = pq.ParquetFile(args.input_parquet)
    neg_samples, pos_samples = calculate_number_of_neg_and_pos_samples(train_val_pq)

    config = create_config(train_file_path=args.input_parquet, test_file_path=args.test_parquet,
                           logs_dir=logs_dir, neg_samples=neg_samples, pos_samples=pos_samples,
                           config=args.config_path)

    training_pipeline = TrainingPipeline(config,
                                         debug=args.debug,
                                         rng=np.random.default_rng(seed=42))

    training_pipeline.run()

    evaluate_validation_set(config, args.input_parquet, args.debug)

    evaluate_test_set(config, args.test_parquet, args.debug)


if __name__ == '__main__':
    main()
