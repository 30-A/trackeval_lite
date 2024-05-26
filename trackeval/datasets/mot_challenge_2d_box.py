import os
import csv
import numpy as np
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from ._base_dataset import _BaseDataset
from .. import utils
from ..utils import TrackEvalException

class MotChallenge2DBox(_BaseDataset):
    """Dataset class for simplified MOT Challenge 2D bounding box tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        default_config = {
            'GT_PATH': None,
            'TRACKER_PATH': None,
            'SEQ_LENGTH': None,
            'PRINT_CONFIG': False,
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset with ground truth path, tracking path, and sequence length"""
        super().__init__()
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())

        if not self.config['GT_PATH'] or not self.config['TRACKER_PATH'] or not self.config['SEQ_LENGTH']:
            raise TrackEvalException('GT_PATH, TRACKER_PATH, and SEQ_LENGTH must be provided.')

        self.gt_path = self.config['GT_PATH']
        self.tracker_path = self.config['TRACKER_PATH']
        self.seq_length = int(self.config['SEQ_LENGTH'])

        self.seq_list = ['sequence']
        self.tracker_list = ['tracker']

    def _load_raw_file(self, tracker, seq, is_gt):
        file_path = self.gt_path if is_gt else self.tracker_path

        read_data, ignore_data = self._load_simple_text_file(file_path)
        num_timesteps = self.seq_length
        data_keys = ['ids', 'classes', 'dets']
        if is_gt:
            data_keys += ['gt_crowd_ignore_regions', 'gt_extras']
        else:
            data_keys += ['tracker_confidences']
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        current_time_keys = [str(t + 1) for t in range(num_timesteps)]
        for t in range(num_timesteps):
            time_key = str(t + 1)
            if time_key in read_data.keys():
                try:
                    time_data = np.asarray(read_data[time_key], dtype=np.float64)
                except ValueError:
                    raise TrackEvalException(f'Cannot convert data for sequence {seq} to float. Is data corrupted?')

                if time_data.shape[1] == 6:
                    if is_gt:
                        default_values = np.array([1, 1, 1])
                    else:
                        default_values = np.array([-1, -1, -1, -1])
                    time_data = np.hstack((time_data, np.tile(default_values, (time_data.shape[0], 1))))

                raw_data['dets'][t] = np.atleast_2d(time_data[:, 2:6])
                raw_data['ids'][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                raw_data['classes'][t] = np.ones_like(raw_data['ids'][t]).astype(int)
                if is_gt:
                    gt_extras_dict = {'zero_marked': np.atleast_1d(time_data[:, 6].astype(int))}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.atleast_1d(time_data[:, 6])
            else:
                raw_data['dets'][t] = np.empty((0, 4))
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['classes'][t] = np.empty(0).astype(int)
                if is_gt:
                    gt_extras_dict = {'zero_marked': np.empty(0)}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.empty(0)
            if is_gt:
                raw_data['gt_crowd_ignore_regions'][t] = np.empty((0, 4))

        if is_gt:
            key_map = {'ids': 'gt_ids', 'classes': 'gt_classes', 'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids', 'classes': 'tracker_classes', 'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq
        return raw_data

    def get_preprocessed_seq_data(self, raw_data):
        self._check_unique_ids(raw_data)

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):
            gt_ids = raw_data['gt_ids'][t]
            gt_dets = raw_data['gt_dets'][t]
            gt_classes = raw_data['gt_classes'][t]
            gt_zero_marked = raw_data['gt_extras'][t]['zero_marked']

            tracker_ids = raw_data['tracker_ids'][t]
            tracker_dets = raw_data['tracker_dets'][t]
            tracker_classes = raw_data['tracker_classes'][t]
            tracker_confidences = raw_data['tracker_confidences'][t]
            similarity_scores = raw_data['similarity_scores'][t]

            data['tracker_ids'][t] = tracker_ids
            data['tracker_dets'][t] = tracker_dets
            data['tracker_confidences'][t] = tracker_confidences
            data['similarity_scores'][t] = similarity_scores

            gt_to_keep_mask = np.equal(gt_classes, 1)
            data['gt_ids'][t] = gt_ids[gt_to_keep_mask]
            data['gt_dets'][t] = gt_dets[gt_to_keep_mask, :]
            data['similarity_scores'][t] = similarity_scores[gt_to_keep_mask]

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int64)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int64)

        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']

        self._check_unique_ids(data, after_preproc=True)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='xywh')
        return similarity_scores