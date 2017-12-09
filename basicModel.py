from features import Feature
from scipy.optimize import minimize
from history import History
from consts import Consts

import numpy as np
from math import exp, log
from multiprocessing.pool import Pool


class BasicModel(object):
    feature = None
    features_idx = None
    jobs_pool = None

    v_parameter = None
    # TODO: find which lambda should we use
    lambda_value = 2

    # Calculates for L
    # For each pair (h, t) holds a list of featurs' idxs it applies
    history_tag_features = None

    def __init__(self, file_full_name: str=Consts.PATH_TO_TRAINING):
        self.feature = Feature(file_full_name, ("100", "103", "104"))

        # Updates 'history_tag_features'
        self._calculate_history_tag_features()
        # Updates 'features_idx'
        self._calculate_gradients_idxs()

        self.jobs_pool = Pool()
        Consts.print_status("minimize", "Computing v_parameter")
        optimize_result = minimize(self._L, np.zeros(len(self.feature.features_occurrences)),
                                   jac=self._gradient, method="L-BFGS-B")
        self.v_parameter = optimize_result.x

    def _calculate_history_tag_features(self):
        Consts.print_status("_calculate_history_tag_features", "Preprocessing")
        self.history_tag_features = {}
        for history in self.feature.histories:
            for tag in Consts.POS_TAGS:
                self.history_tag_features[(history, tag)] = \
                    self.feature.history_matched_features(history, tag)

    def _calculate_gradients_idxs(self):
        Consts.print_status("_calculate_gradients_idxs", "Preprocessing")
        self.features_idx = self.feature.features_occurrences.keys()

    # Calculate v sum in the idx where the feature applies for the pair: (h, t)
    def _calculate_v_sum(self, v_parameter, history: History, tag: str) -> float:
        v_sum = 0
        for idx in self.history_tag_features[(history, tag)]:
            v_sum += v_parameter[idx]
        return v_sum

    # Calculates the sum: sum(exp(v*f(h, t)))
    def _calculate_inner_sum(self, v_parameter, history: History) -> float:
        inner_sum = 0
        for tag in Consts.POS_TAGS:
            inner_sum += exp(self._calculate_v_sum(v_parameter, history, tag))
        return inner_sum

    def v_squares(self, v_parameter):
        v_squares = 0
        for idx in range(0, len(v_parameter)):
            v_squares += v_parameter[idx]**2
        return v_squares

    def _L(self, v_parameter) -> (float, list):
        Consts.print_status("_L", "Calculating")
        left_sum = sum(np.asarray(v_parameter) *
                       np.asarray(list(self.feature.features_occurrences.values())))

        right_sum = 0
        for history in self.feature.histories:
            right_sum += log(self._calculate_inner_sum(v_parameter, history))

        return -(left_sum - right_sum - (self.lambda_value/2)*self.v_squares(v_parameter))

    def _gradient(self, v_parameter):
        Consts.print_status("_gradient", "Calculating")
        return list(map(lambda x: self._derivative_k(v_parameter, x), self.features_idx))

    def _derivative_k(self, v_parameter, k: int) -> float:
        feature_sum = self.feature.features_occurrences[k]

        histories_sum = 0
        for history in self.feature.histories:
            for tag in Consts.POS_TAGS:
                if k in self.history_tag_features[(history, tag)]:
                    histories_sum += self._calculate_v_sum(v_parameter, history, tag) / \
                                     self._calculate_inner_sum(v_parameter, history)

        return -(feature_sum - histories_sum - self.lambda_value*v_parameter[k])

    # Calculates log(p(y|x;v))
    def log_probability(self, history: History, tag: str) -> float:
        return self._calculate_v_sum(self.v_parameter, history, tag) - \
               log(self._calculate_inner_sum(self.v_parameter, history))
