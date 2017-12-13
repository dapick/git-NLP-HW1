from features import Feature
from scipy.optimize import minimize
from history import History
from consts import Consts

import numpy as np
from math import log, exp

from time import time


class BasicModel(object):
    feature = None
    features_amount = None
    features_occurrences_ndarray = None

    v_parameter = None
    # TODO: find which lambda should we use
    lambda_value = 2

    # Common values between iterations

    # Dict of: {(h, t): e^(v*f(h, t))}
    exp_per_history_tag = None
    # Dict of: {h: sum of all the values in 'v_sum_per_history_tag' for h}
    # e.g. for each history saves the sum(e^(v*f(h, t)) for all t in tags
    inner_sum = None

    # Dict of: {(h,t): Calculates e^(v*f(h, t)/sum t in T:(e^(v*f(h, t)) for every (h, t)
    histories_sum = None

    def __init__(self, method: str, file_full_name: str=Consts.PATH_TO_TRAINING):
        self.exp_per_history_tag = {}
        self.inner_sum = {}

        if method == Consts.TRAIN:
            self._training(file_full_name)
        else:
            if method == Consts.TAG:
                self._set_internal_values()

    def _training(self, file_full_name: str=Consts.PATH_TO_TRAINING):
        self.feature = Feature(Consts.TRAIN, ["100", "103", "104"], file_full_name)
        self.features_amount = len(self.feature.features_occurrences)
        self.features_occurrences_ndarray = np.asarray(self.feature.features_occurrences)
        self.v_parameter = self._calculate_v_parameter()
        self.histories_sum = {}

    def _set_internal_values(self):
        with open("../dataFromTraining/v_as_list", 'r') as f:
            self.v_parameter = np.asarray([float(line.rstrip()) for line in f.readlines()])
        self.feature = Feature(Consts.TAG)

    def _calculate_v_parameter(self):
        Consts.print_info("minimize", "Computing v_parameter")
        optimize_result = minimize(fun=self._L, x0=np.zeros(self.features_amount),
                                   jac=self._gradient, method="L-BFGS-B", options={"disp": True, "maxiter": 400})
        return optimize_result.x

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

    def _v_squares(self, v_parameter):
        # Consts.print_debug("_v_squares", "Calculating")
        return sum(np.square(v_parameter))

    def _L(self, v_parameter) -> float:
        Consts.print_info("_L", "Calculating")
        Consts.TIME = 1
        t1 = time()
        left_sum = sum(v_parameter * self.features_occurrences_ndarray)
        Consts.print_time("left_sum", time() - t1)

        t1 = time()
        for history in self.feature.histories:
            self.inner_sum[history] = self._calculate_inner_sum(v_parameter, history)
        Consts.print_time("inner_right_sum", time() - t1)
        right_sum = sum(np.log(list(self.inner_sum.values())))

        return -(left_sum - right_sum - (self.lambda_value/2) * self._v_squares(v_parameter))

    def _gradient(self, v_parameter):
        Consts.print_info("_gradient_at_once", "Calculating")
        Consts.TIME = 1
        self.histories_sum = np.zeros(self.features_amount)
        t1 = time()
        for history in self.feature.histories:
            inner_sum = self.inner_sum[history]
            for tag in self.feature.tags_per_history[history]:
                exp_per_history_tag = self.exp_per_history_tag[(history, tag)]
                for feature_idx in self.feature.history_tag_features[(history, tag)]:
                    self.histories_sum[feature_idx] += exp_per_history_tag / inner_sum
        Consts.print_time("histories_sum", time() - t1)

        gradient = self.features_occurrences_ndarray - self.histories_sum - (self.lambda_value * v_parameter)
        return gradient*(-1)

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
