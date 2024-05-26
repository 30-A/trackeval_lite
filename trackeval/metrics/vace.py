import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import utils

class VACE(_BaseMetric):
    """Class which implements the VACE metrics."""

    @staticmethod
    def get_default_config():
        """Default class config values"""
        default_config = {
            'THRESHOLD': 0.5,  # Similarity score threshold required for a match. Default 0.5.
            'PRINT_CONFIG': False,  # Whether to print the config information on init. Default: False.
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
        self.integer_fields = ['VACE_IDs', 'VACE_GT_IDs', 'num_non_empty_timesteps']
        self.float_fields = ['STDA', 'ATA', 'FDA', 'SFDA']
        self.fields = self.integer_fields + self.float_fields
        self.summary_fields = ['SFDA', 'ATA']

        self._additive_fields = self.integer_fields + ['STDA', 'FDA']

        self.config = utils.init_config(config, self.get_default_config(), self.get_name())
        self.threshold = float(self.config['THRESHOLD'])

    def eval_sequence(self, data):
        res = {}

        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_id_count = np.zeros(data['num_gt_ids'])
        tracker_id_count = np.zeros(data['num_tracker_ids'])
        both_present_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            matches_mask = np.greater_equal(data['similarity_scores'][t], self.threshold)
            match_idx_gt, match_idx_tracker = np.nonzero(matches_mask)
            potential_matches_count[gt_ids_t[match_idx_gt], tracker_ids_t[match_idx_tracker]] += 1
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[tracker_ids_t] += 1
            both_present_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += 1
        union_count = (gt_id_count[:, np.newaxis]
                       + tracker_id_count[np.newaxis, :]
                       - both_present_count)
        with np.errstate(divide='raise', invalid='raise'):
            temporal_iou = potential_matches_count / union_count
        match_rows, match_cols = linear_sum_assignment(-temporal_iou)
        res['STDA'] = temporal_iou[match_rows, match_cols].sum()
        res['VACE_IDs'] = data['num_tracker_ids']
        res['VACE_GT_IDs'] = data['num_gt_ids']

        non_empty_count = 0
        fda = 0
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            n_g = len(gt_ids_t)
            n_d = len(tracker_ids_t)
            if not (n_g or n_d):
                continue
            non_empty_count += 1
            if not (n_g and n_d):
                continue
            spatial_overlap = data['similarity_scores'][t]
            match_rows, match_cols = linear_sum_assignment(-spatial_overlap)
            overlap_ratio = spatial_overlap[match_rows, match_cols].sum()
            fda += overlap_ratio / (0.5 * (n_g + n_d))
        res['FDA'] = fda
        res['num_non_empty_timesteps'] = non_empty_count

        res.update(self._compute_final_fields(res))
        return res

    @staticmethod
    def _compute_final_fields(additive):
        final = {}
        with np.errstate(invalid='ignore'):
            final['ATA'] = (additive['STDA'] /
                            (0.5 * (additive['VACE_IDs'] + additive['VACE_GT_IDs'])))
            final['SFDA'] = additive['FDA'] / additive['num_non_empty_timesteps']
        return final