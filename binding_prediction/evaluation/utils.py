import os

import numpy as np

from binding_prediction.evaluation.evaluation_pipeline import EvaluationPipeline
from binding_prediction.utils import pretty_print_text


def evaluate_test_set(config, pq_file_path, debug):
    pretty_print_text("Get test predictions")
    test_evaluation_pipeline = EvaluationPipeline(config, prediction_pq_file_path=pq_file_path,
                                                  debug=debug)
    test_evaluation_pipeline.create_kaggle_submission_file()


def evaluate_validation_set(config, pq_file_path, debug):
    pretty_print_text("Get validation metrics")
    with open(os.path.join(config.logs_dir, 'val_indices.npy'), 'rb') as f:
        validation_indices = np.load(f)
    val_evaluation_pipeline = EvaluationPipeline(config, prediction_pq_file_path=pq_file_path,
                                                 debug=debug, prediction_indices=validation_indices)
    predictions = val_evaluation_pipeline.run()
    val_evaluation_pipeline.calculate_metrics(predictions)


def get_predictions(weak_learner_config, pq_file_path, indices_in_shard):
    evaluation_pipeline = EvaluationPipeline(weak_learner_config, pq_file_path,
                                             prediction_indices=indices_in_shard)
    predictions = evaluation_pipeline.run()
    return predictions
