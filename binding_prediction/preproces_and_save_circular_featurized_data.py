import argparse
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from binding_prediction.data_processing.circular_fingerprints import process_row_group


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_parquet', type=str, default='data/two_row_groups.parquet')
    parser.add_argument('--output_dir', type=str, default='data/processed')
    parser.add_argument('--protein_map_path', type=str, default='data/processed/circular_3_2048/train/protein_map.json')
    parser.add_argument('--circular_fingerprint_radius', type=int, default=3)
    parser.add_argument('--circular_fingerprint_length', type=int, default=2048)
    return parser.parse_args()


def main():
    args = parse_args()
    parquet_file_path = args.input_parquet
    parquet_file_name = os.path.basename(parquet_file_path)
    subdirectory = f'circular_{args.circular_fingerprint_radius}_{args.circular_fingerprint_length}'
    if args.protein_map_path is None:
        protein_map = {}
    else:
        with open(args.protein_map_path, 'r') as f:
            protein_map = json.load(f)
    output_dir = os.path.join(args.output_dir, parquet_file_name, subdirectory)
    os.makedirs(output_dir, exist_ok=True)
    num_row_groups = pq.ParquetFile(parquet_file_path).metadata.num_row_groups
    for i in tqdm(range(num_row_groups)):
        print(f"Processing row group {i}")
        start_time = time.time()
        protein_map = process_row_group(parquet_file_path, i, output_dir, protein_map, args.circular_fingerprint_radius,
                                        args.circular_fingerprint_length)
        print(f"Total time: {time.time() - start_time}")


if __name__ == '__main__':
    main()
