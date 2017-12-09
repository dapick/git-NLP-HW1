from features import Feature
from scipy.optimize import minimize
from history import History
import numpy as np
from math import exp, log
from consts import Consts


class BasicModel(object):
    feature = None

    v_parameter = None
    # TODO: find which lambda should we use
    lambda_value = 2

    # Calculates for L
    # For each pair (h, t) holds the featurs' idx it applies
    history_tag_features = None

    def __init__(self, file_full_name: str=Consts.PATH_TO_TRAINING):
        self.feature = Feature(file_full_name, ("100", "103", "104"))
        self.feature.feature_100()
        self.feature.feature_103()
        self.feature.feature_104()

        self.history_tag_features = {}
        self._calculate_L_values()
        optimize_result = minimize(self._L, np.zeros(len(self.feature.features_occurrences)), jac=True, method="L-BFGS-B")
        self.v_parameter = optimize_result.x

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

    def _calculate_L_values(self):
        for history in self.feature.histories:
            for tag in Consts.POS_TAGS:
                self.history_tag_features[(history, tag)] = \
                    self.feature.history_matched_features(history, tag)

    def v_squares(self, v_parameter):
        v_squares = 0
        for idx in range(0, len(v_parameter)):
            v_squares += v_parameter[idx]**2
        return v_squares

    def _L(self, v_parameter) -> (float, list):
        left_sum = sum(np.asarray(v_parameter) *
                       np.asarray(list(self.feature.features_occurrences.values())))

        right_sum = 0
        for history in self.feature.histories:
            right_sum += log(self._calculate_inner_sum(v_parameter, history))
        gradients_list = []
        for idx in range(0, len(self.feature.features_occurrences)):
            gradients_list.append(self._gradient_k(v_parameter, idx))

        return -(left_sum - right_sum - (self.lambda_value/2)*self.v_squares(v_parameter)), gradients_list

    def _gradient_k(self, v_parameter, k: int) -> float:
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
