from ._base_metric import _BaseMetric
from .. import utils

class Count(_BaseMetric):
    """Class which simply counts the number of tracker and gt detections and ids."""

    @staticmethod
    def get_default_config():
        """Default class config values"""
        default_config = {
            'PRINT_CONFIG': False,  # Whether to print the config information on init. Default: False.
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
        self.integer_fields = ['Dets', 'GT_Dets', 'IDs', 'GT_IDs']
        self.fields = self.integer_fields
        self.summary_fields = self.fields
        self.config = utils.init_config(config, self.get_default_config(), self.get_name())  # Initialize config

    def eval_sequence(self, data):
        res = {'Dets': data['num_tracker_dets'],
               'GT_Dets': data['num_gt_dets'],
               'IDs': data['num_tracker_ids'],
               'GT_IDs': data['num_gt_ids'],
               'Frames': data['num_timesteps']}
        return res