from features import Feature
from scipy.optimize import minimize
from history import History
from consts import Consts

import numpy as np
from math import log, exp
from multiprocessing.pool import Pool
from functools import partial

from time import time


class BasicModel(object):
    feature = None
    features_idx = None

    v_parameter = None
    # TODO: find which lambda should we use
    lambda_value = 2

    # Common values between iterations

    # Dict of: {(h, t): e^(v*f(h, t))}
    exp_per_history_tag = None
    # Dict of: {h: sum of all the values in 'v_sum_per_history_tag' for h}
    # e.g. for each history saves the sum(e^(v*f(h, t)) for all t in tags
    inner_sum = None

    def __init__(self, method: str, file_full_name: str=Consts.PATH_TO_TRAINING):
        if method == Consts.TRAIN:
            self._training(file_full_name)
        # else:
        #     if method == Consts.TAG:
        #         self._set_internal_values()

    def _training(self, file_full_name: str=Consts.PATH_TO_TRAINING):
        self.feature = Feature(file_full_name, ("100", "103", "104"))
        self.features_idx = list(self.feature.features_occurrences.keys())
        self.exp_per_history_tag = {}
        self.inner_sum = {}
        self.v_parameter = self._calculate_v_parameter()

    # def _set_internal_values(self):

    def _calculate_v_parameter(self):
        Consts.print_info("minimize", "Computing v_parameter")
        optimize_result = minimize(fun=self._L, x0=np.ones(len(self.feature.features_occurrences)),
                                   jac=self._gradient, method="L-BFGS-B", options={"disp": True, "maxiter": 400})
        return optimize_result.x

    # Calculate v sum in the idx where the feature applies for the pair: (h, t)
    def _calculate_v_sum(self, v_parameter, history: History, tag: str) -> float:
        Consts.print_debug("_calculate_v_sum for: " + history.get_current_word() + ", " + tag, "Calculating")
        list_idx = self.feature.history_tag_features.get((history, tag))
        if list_idx:
            return sum(v_parameter[list_idx])
        return 0

    # Calculates the sum: sum(exp(v*f(h, t)))
    def _calculate_inner_sum(self, v_parameter, history: History) -> float:
        Consts.print_debug("_calculate_inner_sum for: " + history.get_current_word(), "Calculating")
        count = 0
        for tag in self.feature.tags_per_history[history]:
            self.exp_per_history_tag[(history, tag)] = exp(self._calculate_v_sum(v_parameter, history, tag))
            count += self.exp_per_history_tag[(history, tag)]
        return count

    def _v_squares(self, v_parameter):
        Consts.print_debug("_v_squares", "Calculating")
        return sum(np.square(v_parameter))

    def _L(self, v_parameter) -> float:
        Consts.print_info("_L", "Calculating")
        Consts.TIME = 1
        t1 = time()
        left_sum = sum(v_parameter * np.asarray(list(self.feature.features_occurrences.values())))
        Consts.print_time("left_sum", time() - t1)

        t1 = time()
        for history in self.feature.histories:
            self.inner_sum[history] = self._calculate_inner_sum(v_parameter, history)
        Consts.print_time("inner_right_sum" ,time() - t1)
        right_sum = sum(np.log(list(self.inner_sum.values())))

        return -(left_sum - right_sum - (self.lambda_value/2) * self._v_squares(v_parameter))

    def _gradient(self, v_parameter):
        Consts.print_info("_gradient", "Calculating")

        with Pool(6) as jobs_pool:
            gradient = jobs_pool.map(partial(self._derivative_k, v_parameter), self.features_idx)

        return np.asarray(list(gradient))

    def _derivative_k(self, v_parameter, k: int) -> float:
        Consts.print_info("_derivative_" + str(k), "Calculating")
        feature_sum = self.feature.features_occurrences[k]

        histories_sum = 0
        for history in self.feature.histories:
            for tag in self.feature.tags_per_history[history]:
                if k in self.feature.history_tag_features[(history, tag)]:
                    histories_sum += self.exp_per_history_tag[(history, tag)] / \
                                     self.inner_sum[history]

        return -(feature_sum - histories_sum - self.lambda_value*v_parameter[k])

    # Calculates log(p(y|x;v))
    def log_probability(self, history: History, tag: str) -> float:
        # TODO: save calculates for inner_sum per history
        return self._calculate_v_sum(self.v_parameter, history, tag) - \
               log(self._calculate_inner_sum(self.v_parameter, history))
