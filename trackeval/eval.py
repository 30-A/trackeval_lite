import traceback
import os
from . import utils
from .utils import TrackEvalException
from .metrics import Count

class Evaluator:
    """Evaluator class for evaluating different metrics for different datasets"""

    @staticmethod
    def get_default_eval_config():
        """Returns the default config values for evaluation"""
        default_config = {
            'PRINT_CONFIG': False
        }
        return default_config

    def __init__(self, config=None):
        """Initialise the evaluator with a config file"""
        self.config = utils.init_config(config, self.get_default_eval_config(), 'Eval')

    def evaluate(self, dataset_list, metrics_list, show_progressbar=False):
        """Evaluate a set of metrics on a set of datasets"""
        config = self.config
        metrics_list = metrics_list + [Count()]  # Count metrics are always run
        metric_names = utils.validate_metrics_list(metrics_list)
        dataset_names = [dataset.get_name() for dataset in dataset_list]
        output_res = {}
        output_msg = {}

        for dataset, dataset_name in zip(dataset_list, dataset_names):
            # Initialize the output dictionaries for each dataset
            output_res[dataset_name] = {}
            output_msg[dataset_name] = {}  # Ensure this is initialized

            # Get dataset info about what to evaluate
            tracker_list, seq_list = dataset.get_eval_info()

            # Evaluate each tracker
            for tracker in tracker_list:

                # Evaluate the single sequence
                seq = seq_list[0]
                res = {seq: eval_sequence(seq, dataset, tracker, metrics_list, metric_names)}

                for metric, metric_name in zip(metrics_list, metric_names):
                    table_res = {seq: res[seq][metric_name]}
                    metric.print_table(table_res, "", 1)

                # Output for returning from function
                output_res[dataset_name][tracker] = res
                output_msg[dataset_name][tracker] = 'Success'

        return output_res, output_msg

def eval_sequence(seq, dataset, tracker, metrics_list, metric_names):
    """Function for evaluating a single sequence"""

    raw_data = dataset.get_raw_seq_data(tracker, seq)
    seq_res = {}
    data = dataset.get_preprocessed_seq_data(raw_data)
    for metric, met_name in zip(metrics_list, metric_names):
        seq_res[met_name] = metric.eval_sequence(data)
    return seq_res