import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import utils

class CLEAR(_BaseMetric):
    """Class which implements the CLEAR metrics"""

    @staticmethod
    def get_default_config():
        """Default class config values"""
        default_config = {
            'THRESHOLD': 0.5,  # Similarity score threshold required for a TP match. Default 0.5.
            'PRINT_CONFIG': False,  # Whether to print the config information on init. Default: False.
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
        main_integer_fields = ['CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag']
        extra_integer_fields = ['CLR_Frames']
        self.integer_fields = main_integer_fields + extra_integer_fields
        main_float_fields = ['MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'sMOTA']
        extra_float_fields = ['CLR_F1', 'FP_per_frame', 'MOTAL', 'MOTP_sum']
        self.float_fields = main_float_fields + extra_float_fields
        self.fields = self.float_fields + self.integer_fields
        self.summed_fields = self.integer_fields + ['MOTP_sum']
        self.summary_fields = main_float_fields + main_integer_fields

        self.config = utils.init_config(config, self.get_default_config(), self.get_name())
        self.threshold = float(self.config['THRESHOLD'])

    def eval_sequence(self, data):
        res = {}
        for field in self.fields:
            res[field] = 0

        if data['num_tracker_dets'] == 0:
            res['CLR_FN'] = data['num_gt_dets']
            res['ML'] = data['num_gt_ids']
            res['MLR'] = 1.0
            return res
        if data['num_gt_dets'] == 0:
            res['CLR_FP'] = data['num_tracker_dets']
            res['MLR'] = 1.0
            return res

        num_gt_ids = data['num_gt_ids']
        gt_id_count = np.zeros(num_gt_ids)
        gt_matched_count = np.zeros(num_gt_ids)
        gt_frag_count = np.zeros(num_gt_ids)

        prev_tracker_id = np.nan * np.zeros(num_gt_ids)
        prev_timestep_tracker_id = np.nan * np.zeros(num_gt_ids)

        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            if len(gt_ids_t) == 0:
                res['CLR_FP'] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                res['CLR_FN'] += len(gt_ids_t)
                gt_id_count[gt_ids_t] += 1
                continue

            similarity = data['similarity_scores'][t]
            score_mat = (tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]])
            score_mat = 1000 * score_mat + similarity
            score_mat[similarity < self.threshold - np.finfo('float').eps] = 0

            match_rows, match_cols = linear_sum_assignment(-score_mat)
            actually_matched_mask = score_mat[match_rows, match_cols] > 0 + np.finfo('float').eps
            match_rows = match_rows[actually_matched_mask]
            match_cols = match_cols[actually_matched_mask]

            matched_gt_ids = gt_ids_t[match_rows]
            matched_tracker_ids = tracker_ids_t[match_cols]

            prev_matched_tracker_ids = prev_tracker_id[matched_gt_ids]
            is_idsw = (np.logical_not(np.isnan(prev_matched_tracker_ids))) & (
                np.not_equal(matched_tracker_ids, prev_matched_tracker_ids))
            res['IDSW'] += np.sum(is_idsw)

            gt_id_count[gt_ids_t] += 1
            gt_matched_count[matched_gt_ids] += 1
            not_previously_tracked = np.isnan(prev_timestep_tracker_id)
            prev_tracker_id[matched_gt_ids] = matched_tracker_ids
            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids
            currently_tracked = np.logical_not(np.isnan(prev_timestep_tracker_id))
            gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)

            num_matches = len(matched_gt_ids)
            res['CLR_TP'] += num_matches
            res['CLR_FN'] += len(gt_ids_t) - num_matches
            res['CLR_FP'] += len(tracker_ids_t) - num_matches
            if num_matches > 0:
                res['MOTP_sum'] += sum(similarity[match_rows, match_cols])

        tracked_ratio = gt_matched_count[gt_id_count > 0] / gt_id_count[gt_id_count > 0]
        res['MT'] = np.sum(np.greater(tracked_ratio, 0.8))
        res['PT'] = np.sum(np.greater_equal(tracked_ratio, 0.2)) - res['MT']
        res['ML'] = num_gt_ids - res['MT'] - res['PT']
        res['Frag'] = np.sum(np.subtract(gt_frag_count[gt_frag_count > 0], 1))
        res['MOTP'] = res['MOTP_sum'] / np.maximum(1.0, res['CLR_TP'])

        res['CLR_Frames'] = data['num_timesteps']

        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res):
        num_gt_ids = res['MT'] + res['ML'] + res['PT']
        res['MTR'] = res['MT'] / np.maximum(1.0, num_gt_ids)
        res['MLR'] = res['ML'] / np.maximum(1.0, num_gt_ids)
        res['PTR'] = res['PT'] / np.maximum(1.0, num_gt_ids)
        res['CLR_Re'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['CLR_Pr'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res['CLR_FP'])
        res['MODA'] = (res['CLR_TP'] - res['CLR_FP']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['MOTA'] = (res['CLR_TP'] - res['CLR_FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['MOTP'] = res['MOTP_sum'] / np.maximum(1.0, res['CLR_TP'])
        res['sMOTA'] = (res['MOTP_sum'] - res['CLR_FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])

        res['CLR_F1'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + 0.5 * res['CLR_FN'] + 0.5 * res['CLR_FP'])
        res['FP_per_frame'] = res['CLR_FP'] / np.maximum(1.0, res['CLR_Frames'])
        safe_log_idsw = np.log10(res['IDSW']) if res['IDSW'] > 0 else res['IDSW']
        res['MOTAL'] = (res['CLR_TP'] - res['CLR_FP'] - safe_log_idsw) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        return res
