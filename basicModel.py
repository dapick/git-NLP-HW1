from features import Feature
from history import History
from consts import Consts
from training import Training

import numpy as np
from math import log
import pickle


class BasicModel(object):
    feature = None

    v_parameter = None

    # Common values between iterations

    # Dict of: {h: ndarrag of v*f(h, t) for every possible tag}
    inner_sum = None

    def __init__(self, method: str, file_full_name: str=Consts.PATH_TO_TRAINING):
        if method == Consts.TRAIN:
            self._training(file_full_name)
        elif method == Consts.TAG:
            self._set_internal_values()

    def _training(self, file_full_name: str=Consts.PATH_TO_TRAINING):
        self.v_parameter = Training(Consts.BASIC_MODEL, ["100", "103", "104"], file_full_name).v_parameter
        with open("../data_from_training/basic_model/v_parameter", 'wb') as f:
            pickle.dump(self.v_parameter, f, protocol=-1)

    def _set_internal_values(self):
        self.inner_sum = {}
        # with open("../data_from_training/basic_model/v_as_list", 'r') as f:
        #     self.v_parameter = np.asarray([float(line.rstrip()) for line in f.readlines()])
        with open("../data_from_training/basic_model/v_parameter", 'rb') as f:
            self.v_parameter = pickle.load(f)
        self.feature = Feature(Consts.TAG, Consts.BASIC_MODEL)

    def _get_applied_features(self, history: History, tag: str) -> list:
        list_idxs = self.feature.history_tag_features.get((history, tag))
        if list_idxs is None:
            list_idxs = self.feature.history_matched_features(history, tag)
        return list_idxs

    # Calculates the sum: sum(exp(v*f(h, t)))
    def _calculate_inner_sum(self, history: History) -> float:
        inner_sum = self.inner_sum.get(history)
        if inner_sum is None:
            features_on_history = [self._get_applied_features(history, tag) for tag in Consts.POS_TAGS]
            inner_sum = np.asarray(
                [sum(self.v_parameter[features_idxs]) for features_idxs in features_on_history])
            self.inner_sum[history] = inner_sum
        return sum(np.exp(inner_sum))

    # Calculates log(p(y|x;v))
    def log_probability(self, history: History, tag_idx: int) -> float:
        inner_sum = self._calculate_inner_sum(history)
        v_sum = self.inner_sum[history][tag_idx]

        return v_sum - log(inner_sum)
