from features import Feature
from scipy.optimize import minimize
from history import History
from consts import Consts

import numpy as np
from math import log
from multiprocessing.pool import Pool
from functools import partial


class BasicModel(object):
    feature = None
    features_idx = None

    v_parameter = None
    # TODO: find which lambda should we use
    lambda_value = 2

    def __init__(self, file_full_name: str=Consts.PATH_TO_TRAINING):
        self.feature = Feature(file_full_name, ("100", "103", "104"))

        # Updates 'features_idx'
        self._calculate_gradients_idxs()

        Consts.print_info("minimize", "Computing v_parameter")
        v_start_value = np.zeros(len(self.feature.features_occurrences))
        optimize_result = minimize(fun=self._L, x0=v_start_value,
                                   jac=self._gradient, method="L-BFGS-B", options={"disp": True})
        self.v_parameter = optimize_result.x
        Consts.DEBUG = 1

    def _calculate_gradients_idxs(self):
        Consts.print_info("_calculate_gradients_idxs", "Preprocessing")
        self.features_idx = list(self.feature.features_occurrences.keys())

    # Calculate v sum in the idx where the feature applies for the pair: (h, t)
    def _calculate_v_sum(self, v_parameter, history: History, tag: str) -> float:
        Consts.print_debug("_calculate_v_sum for: " + history.get_current_word() + ", " + tag, "Calculating")
        list_idx = self.feature.history_tag_features.get((history, tag))
        if list_idx:
            return sum(v_parameter[list_idx])
        return 0
        # return sum(v_parameter[self.feature.history_tag_features.get((history, tag))])

    # Calculates the sum: sum(exp(v*f(h, t)))
    def _calculate_inner_sum(self, v_parameter, history: History) -> float:
        Consts.print_debug("_calculate_inner_sum for: " + history.get_current_word(), "Calculating")
        v_sums = [self._calculate_v_sum(v_parameter, history, tag) for tag in Consts.POS_TAGS]
        return sum(np.exp(v_sums))

    def _v_squares(self, v_parameter):
        Consts.print_debug("_v_squares", "Calculating")
        return sum(np.square(v_parameter))

    def _L(self, v_parameter) -> (float, list):
        Consts.print_info("_L", "Calculating")
        left_sum = sum(v_parameter * np.asarray(list(self.feature.features_occurrences.values())))

        inner_right_sum = [self._calculate_inner_sum(v_parameter, history) for history in self.feature.histories]
        right_sum = sum(np.log(inner_right_sum))

        return -(left_sum - right_sum - (self.lambda_value/2) * self._v_squares(v_parameter))

    def _gradient(self, v_parameter):
        Consts.print_info("_gradient", "Calculating")

        with Pool(4) as jobs_pool:
            gradient = jobs_pool.map(partial(self._derivative_k, v_parameter), self.features_idx)

        return np.asarray(list(gradient))

    def _derivative_k(self, v_parameter, k: int) -> float:
        Consts.print_info("_derivative_" + str(k), "Calculating")
        feature_sum = self.feature.features_occurrences[k]

        histories_sum = 0
        # for history in self.feature.histories:
        #     for tag in Consts.POS_TAGS:
        #         if k in self.feature.history_tag_features[(history, tag)]:
        #             histories_sum += self._calculate_v_sum(v_parameter, history, tag) / \
        #                              self._calculate_inner_sum(v_parameter, history)
        for (history, tag) in self.feature.history_tag_features.keys():
            if k in self.feature.history_tag_features[(history, tag)]:
                histories_sum += self._calculate_v_sum(v_parameter, history, tag) / \
                                 self._calculate_inner_sum(v_parameter, history)

        return -(feature_sum - histories_sum - self.lambda_value*v_parameter[k])

    # Calculates log(p(y|x;v))
    def log_probability(self, history: History, tag: str) -> float:
        return self._calculate_v_sum(self.v_parameter, history, tag) - \
               log(self._calculate_inner_sum(self.v_parameter, history))
