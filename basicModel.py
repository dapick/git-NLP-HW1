from features import Feature
from history import History
from consts import Consts
from training import Training

import numpy as np
from math import log, exp
import pickle


class BasicModel(object):
    feature = None

    v_parameter = None

    # Common values between iterations

    # Dict of: {(h, t): e^(v*f(h, t))}
    exp_per_history_tag = None
    # Dict of: {h: sum of all the values in 'v_sum_per_history_tag' for h}
    # e.g. for each history saves the sum(e^(v*f(h, t)) for all t in tags
    inner_sum = None

    def __init__(self, method: str, file_full_name: str=Consts.PATH_TO_TRAINING):
        if method == Consts.TRAIN:
            self._training(file_full_name)
        else:
            if method == Consts.TAG:
                self._set_internal_values()

    def _training(self, file_full_name: str=Consts.PATH_TO_TRAINING):
        self.v_parameter = Training(Consts.BASIC_MODEL, ["100", "103", "104"], file_full_name).v_parameter
        with open("../data_from_training/basic_model/v_parameter", 'wb') as f:
            pickle.dump(self.v_parameter, f, protocol=-1)

    def _set_internal_values(self):
        self.exp_per_history_tag = {}
        self.inner_sum = {}
        with open("../data_from_training/basic_model/v_as_list", 'r') as f:
            self.v_parameter = np.asarray([float(line.rstrip()) for line in f.readlines()])
        # TODO: Uncomment after running evaluation on v again on train.wtag
        # with open("../data_from_training/basic_model/v_parameter", 'rb') as f:
        #     self.v_parameter = pickle.load(f)
        self.feature = Feature(Consts.TAG, Consts.BASIC_MODEL)

    # Calculate v sum in the idx where the feature applies for the pair: (h, t)
    def _calculate_v_sum(self, history: History, tag: str) -> float:
        # Consts.print_debug("_calculate_v_sum for: " + history.get_current_word() + ", " + tag, "Calculating")
        list_idx = self.feature.history_tag_features.get((history, tag))
        if not list_idx:
            list_idx = self.feature.history_matched_features(history, tag)
        if list_idx:
            return sum(self.v_parameter[list_idx])
        return 0

    # Calculates the sum: sum(exp(v*f(h, t)))
    def _calculate_inner_sum(self, history: History) -> float:
        # Consts.print_debug("_calculate_inner_sum for: " + history.get_current_word(), "Calculating")
        count = 0
        for tag in Consts.POS_TAGS:
            v_sum = self.exp_per_history_tag.get((history, tag))
            if not v_sum:
                v_sum = exp(self._calculate_v_sum(history, tag))
                self.exp_per_history_tag[(history, tag)] = v_sum
            count += v_sum
        return count

    # Calculates log(p(y|x;v))
    def log_probability(self, history: History, tag: str) -> float:
        v_sum = self.exp_per_history_tag.get((history, tag))
        if v_sum:
            v_sum = log(v_sum)
        else:
            v_sum = self._calculate_v_sum(history, tag)
            self.exp_per_history_tag[(history, tag)] = exp(v_sum)

        inner_sum = self.inner_sum.get(history)
        if not inner_sum:
            inner_sum = self._calculate_inner_sum(history)
            self.inner_sum[history] = inner_sum

        return v_sum - log(inner_sum)
