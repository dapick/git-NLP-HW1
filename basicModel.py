from features import Feature
from scipy.optimize import minimize
from history import Histories, History
import numpy as np
from math import exp
from tags import Tags


class BasicModel(object):
    features_vector = None
    histories = None
    tags = None
    v_parameter = 0
    # TODO: find which lambda should we use
    lambda_value = 2

    def __init__(self):
        self.features_vector = [Feature.feature_100,
                                Feature.feature_103,
                                Feature.feature_104]
        self.histories, self.tags = \
            Histories.build_history_list_and_tags_list("data/train.wtag")

    # TODO: decide if v_parameter is a parameter for this function or not
    # TODO: convert to the log version
    def probability(self, history: History, tag: str):
        f_value_np_array = np.asarray([
                    self.features_vector[0](history, tag),
                    self.features_vector[1](history, tag),
                    self.features_vector[2](history, tag)])
        numerator = exp(np.sum(self.v_parameter * f_value_np_array))

        denominator = 0
        for curr_tag in Tags.POS_TAGS:
            f_value_np_array = np.asarray([
                    self.features_vector[0](history, curr_tag),
                    self.features_vector[1](history, curr_tag),
                    self.features_vector[2](history, curr_tag)])
            denominator += exp(np.sum(self.v_parameter * f_value_np_array))

        return numerator/denominator

    # TODO: decide if v_parameter is a parameter for this function or not
    def gradient_k(self, k: int):
        # Calculate the first argument of the gradient
        feature_sum = 0
        for history, tag in zip(self.histories, self.tags):
            feature_sum += self.features_vector[k](history, tag)

        # Calculate the second argument of the gradient
        histories_sum = 0
        for history in self.histories:
            history_sum = 0
            for tag in self.tags:
                history_sum += \
                    self.features_vector[k](history, tag) * self.probability(history, tag)
            history_sum += history_sum

        return feature_sum - histories_sum





