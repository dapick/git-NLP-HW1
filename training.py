from consts import Consts
from features import Feature
from history import History

from scipy.optimize import minimize
import numpy as np
from math import exp
from time import time


class Training(object):
    feature = None
    features_amount = None
    features_occurrences_ndarray = None

    v_parameter = None
    # TODO: find which lambda should we use
    lambda_value = None

    # Common values between iterations

    # Dict of: {(h, t): e^(v*f(h, t))}
    exp_per_history_tag = None
    # Dict of: {h: sum of all the values in 'v_sum_per_history_tag' for h}
    # e.g. for each history saves the sum(e^(v*f(h, t)) for all t in tags
    inner_sum = None

    # Dict of: {(h,t): Calculates e^(v*f(h, t)/sum t in T:(e^(v*f(h, t)) for every (h, t)
    histories_sum = None

    def __init__(self, model: str, used_features: list, lambda_value: float, file_full_name: str=Consts.PATH_TO_TRAINING):
        self.feature = Feature(Consts.TRAIN, model, used_features, file_full_name)
        self.features_amount = len(self.feature.features_occurrences)
        self.features_occurrences_ndarray = np.asarray(self.feature.features_occurrences)
        self.exp_per_history_tag = {}
        self.inner_sum = {}
        self.lambda_value = lambda_value
        self.iterate_number = 0
        self.time_started_LBFGS = 0
        self.v_parameter = self._calculate_v_parameter()

    def _calculate_v_parameter(self):
        Consts.print_info("minimize", "Computing v_parameter")
        # For seeing the whole process of LBFGS add '"disp": True' to the 'options' dict
        t1 = time()
        optimize_result = minimize(fun=self._L, x0=np.zeros(self.features_amount),
                                   jac=self._gradient, method="L-BFGS-B", options={"maxiter": 400})
        Consts.TIME = 1
        Consts.print_time("_calculate_v_parameter", time() - t1)
        return optimize_result.x

    # Calculate v sum in the idx where the feature applies for the pair: (h, t)
    def _calculate_v_sum(self, v_parameter, history: History, tag: str) -> float:
        # Consts.print_debug("_calculate_v_sum for: " + history.get_current_word() + ", " + tag, "Calculating")
        list_idx = self.feature.history_tag_features.get((history, tag))
        if list_idx:
            return sum(v_parameter[list_idx])
        return 0

    # Calculates the sum: sum(exp(v*f(h, t)))
    def _calculate_inner_sum(self, v_parameter, history: History) -> float:
        # Consts.print_debug("_calculate_inner_sum for: " + history.get_current_word(), "Calculating")
        count = 0
        for tag in self.feature.tags_per_history[history]:
            exp_per_history_tag = exp(self._calculate_v_sum(v_parameter, history, tag))
            count += exp_per_history_tag
            self.exp_per_history_tag[(history, tag)] = exp_per_history_tag
        return count

    def _v_squares(self, v_parameter):
        # Consts.print_debug("_v_squares", "Calculating")
        return sum(np.square(v_parameter))

    def _L(self, v_parameter) -> float:
        # Consts.print_info("_L", "Calculating")
        self.time_started_LBFGS = time()

        left_sum = sum(v_parameter * self.features_occurrences_ndarray)

        for history in self.feature.histories:
            self.inner_sum[history] = self._calculate_inner_sum(v_parameter, history)
        right_sum = sum(np.log(list(self.inner_sum.values())))

        return -(left_sum - right_sum - (self.lambda_value/2) * self._v_squares(v_parameter))

    def _gradient(self, v_parameter):
        # Consts.print_info("_gradient", "Calculating")
        # t1 = time()

        self.histories_sum = np.zeros(self.features_amount)
        for history in self.feature.histories:
            inner_sum = self.inner_sum[history]
            for tag in self.feature.tags_per_history[history]:
                exp_per_history_tag = self.exp_per_history_tag[(history, tag)]
                for feature_idx in self.feature.history_tag_features[(history, tag)]:
                    self.histories_sum[feature_idx] += exp_per_history_tag / inner_sum

        gradient = self.features_occurrences_ndarray - self.histories_sum - (self.lambda_value * v_parameter)

        Consts.TIME = 1
        Consts.print_time("Iterate number " + str(self.iterate_number), time() - self.time_started_LBFGS)
        self.iterate_number += 1

        return gradient*(-1)
