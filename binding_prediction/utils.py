import time

import numpy as np


def get_indices_in_shard(indices, current_shard_num, shard_size):
    indices_in_shard = np.array(indices[np.where(
        (indices >= current_shard_num * shard_size) & (
                indices < (current_shard_num + 1) * shard_size))])
    relative_indices = indices_in_shard - current_shard_num * shard_size
    return indices_in_shard, relative_indices


def calculate_number_of_neg_and_pos_samples(pq_file,
                                            pq_groups_numbers=None,
                                            indices=None):
    neg_samples = 0
    pos_samples = 0
    if pq_groups_numbers is None:
        pq_groups_numbers = range(pq_file.metadata.num_row_groups)
    shard_size = pq_file.metadata.row_group(0).num_rows
    for group_id in pq_groups_numbers:
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
    print(f"{'=' * len(text)}\n{text.upper()}\n{'=' * len(text)}")
