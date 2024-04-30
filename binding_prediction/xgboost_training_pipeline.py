import argparse
import json
import os
import pickle
import time

import numpy as np
import pyarrow.parquet as pq
import xgboost

from binding_prediction.datasets.xgboost_iterator import SmilesIterator
from binding_prediction.evaluation.kaggle_submission_creation import get_submission_test_predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_parquet', type=str, default='data/two_row_groups.parquet')
    parser.add_argument('--test_parquet', type=str, default='data/test.parquet')
    parser.add_argument('--featurizer', type=str, default='circular')
    parser.add_argument('--circular_fingerprint_radius', type=int, default=3)
    parser.add_argument('--circular_fingerprint_length', type=int, default=2048)
    parser.add_argument('--debug', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    current_date = time.strftime("%Y-%m-%d_%H-%M-%S")
    logs_dir = os.path.join('logs', current_date)
    os.makedirs(logs_dir, exist_ok=True)
    print('Train validation split')
    start_time = time.time()
    train_file_path = args.input_parquet
    rng = np.random.default_rng(seed=42)
    train_val_pq = pq.ParquetFile(train_file_path)

    neg_samples = 0
    pos_samples = 0
    for group_id in range(train_val_pq.metadata.num_row_groups):
        group_df = train_val_pq.read_row_group(group_id).to_pandas()
        neg_samples += len(group_df[group_df['binds'] == 0])
        pos_samples += len(group_df[group_df['binds'] == 1])

    if args.debug:
        train_size = 100000
    else:
        train_size = train_val_pq.metadata.num_rows
    train_val_indices = rng.choice(train_val_pq.metadata.num_rows,
                                   train_size,
                                   replace=False)
    train_indices = rng.choice(train_val_indices, int(0.5 * train_size), replace=False)
    val_indices = np.setdiff1d(train_val_indices, train_indices)
    print(f"Train validation split time: {time.time() - start_time}")

    print('Creating datasets')
    start_time = time.time()
    train_dataset = SmilesIterator(train_file_path, indicies=train_indices,
                                   fingerprint=args.featurizer,
                                   radius=args.circular_fingerprint_radius,
                                   nBits=args.circular_fingerprint_length)
    val_dataset = SmilesIterator(train_file_path, indicies=val_indices,
                                 fingerprint=args.featurizer,
                                 radius=args.circular_fingerprint_radius,
                                 nBits=args.circular_fingerprint_length)
    if os.path.exists(os.path.join(train_dataset._cache_path, "train_Xy.pkl")):
        with open(os.path.join(train_dataset._cache_path, "train_Xy.pkl"), 'rb') as file:
            train_Xy = pickle.load(file)
    else:
        train_Xy = xgboost.DMatrix(train_dataset)
        with open(os.path.join(train_dataset._cache_path, 'train_Xy.pkl'), 'wb') as file:
            pickle.dump(train_Xy, file)
    if os.path.exists(os.path.join(val_dataset._cache_path, "val_Xy.pkl")):
        with open(os.path.join(val_dataset._cache_path, "val_Xy.pkl"), 'rb') as file:
            val_Xy = pickle.load(file)
    else:
        val_Xy = xgboost.DMatrix(val_dataset)
        with open(os.path.join(val_dataset._cache_path, 'val_Xy.pkl'), 'wb') as file:
            pickle.dump(val_Xy, file)

    print(f"Creating datasets time: {time.time() - start_time}")
    print(f"Datasets sizes: train: {train_Xy.num_row()}, "
          f"validation: {val_Xy.num_row()}")
    print('Creating model')
    start_time = time.time()
    params = {
        'eta': 0.1,
        'gamma': 0.1,
        'max_depth': 15,
        'objective': 'binary:logistic',
        'eval_metric': 'map',
        'verbosity': 2,
        'nthread': 12,
        "tree_method": "hist",
        "grow_policy": 'depthwise',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': neg_samples / pos_samples,
    }
    with open(os.path.join(logs_dir, 'params.json'), 'w') as file:
        json.dump(params, file)
    num_rounds = 100

    eval_list = [(train_Xy, 'train'), (val_Xy, 'eval')]
    print(f"Creating model time: {time.time() - start_time}")
    print('Training model')
    start_time = time.time()
    try:
        model = xgboost.train(params, train_Xy, num_rounds,
                              evals=eval_list, verbose_eval=True,
                              early_stopping_rounds=5
                              )
    except KeyboardInterrupt:
        print('Training interrupted')
    print(f"Training model time: {time.time() - start_time}")
    print('Saving model')
    start_time = time.time()
    with open(os.path.join(logs_dir, 'model.pkl'), 'wb') as file:
        pickle.dump(model, file)
    print(f"Saving model time: {time.time() - start_time}")

    print('Testing model')
    start_time = time.time()
    test_dataset = SmilesIterator(args.test_parquet, shuffle=False,
                                  fingerprint=args.featurizer,
                                  radius=args.circular_fingerprint_radius,
                                  nBits=args.circular_fingerprint_length)
    test_Xy = xgboost.DMatrix(test_dataset)
    print(f"Load test data: {time.time() - start_time}")
    get_submission_test_predictions(test_dataset, test_Xy,
                                    model, logs_dir)
    print('Done')


if __name__ == '__main__':
    main()
