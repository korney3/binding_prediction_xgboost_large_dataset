import time
from dataclasses import dataclass


def calculate_number_of_neg_and_pos_samples(pq_file):
    neg_samples = 0
    pos_samples = 0
    for group_id in range(pq_file.metadata.num_row_groups):
        group_df = pq_file.read_row_group(group_id).to_pandas()
        neg_samples += len(group_df[group_df['binds'] == 0])
        pos_samples += len(group_df[group_df['binds'] == 1])
    return neg_samples, pos_samples


@dataclass
class ModelTypes:
    XGBOOST = 'xgboost'


@dataclass
class FeaturizerTypes:
    CIRCULAR = 'circular_fingerprint'


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
