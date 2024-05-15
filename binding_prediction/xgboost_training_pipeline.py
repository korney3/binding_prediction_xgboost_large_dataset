import argparse
import os
import time

import numpy as np
import pyarrow.parquet as pq

from binding_prediction.config.config import create_config
from binding_prediction.evaluation.utils import evaluate_test_set, evaluate_validation_set
from binding_prediction.runner import Runner
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

    runner = Runner(train_parquet_path=args.input_parquet,
                    test_parquet_path=args.test_parquet,
                    config_path=args.config_path, debug=args.debug,
                    logs_dir_location="logs", seed=42)
    runner.run()


if __name__ == '__main__':
    main()
