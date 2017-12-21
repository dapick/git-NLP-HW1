from consts import Consts
from features import Feature

from scipy.optimize import minimize
from scipy.special import logsumexp
import numpy as np
from time import time


class Training(object):
    feature = None
    features_occurrences_ndarray = None

    v_parameter = None
    # lambda_value = None

    # Common values between iterations
    right_sum_for_L = None
    product_for_gradient = None

    def __init__(self, model: str, used_features: list, lambda_value: float, file_full_name: str):
        self.feature = Feature(Consts.TRAIN, model, used_features, file_full_name)
        self.features_occurrences_ndarray = np.asarray(self.feature.features_occurrences)
        self.lambda_value = np.array([lambda_value])

        self.iterate_number = 0
        self.time_started_LBFGS = 0
        self.v_parameter = self._calculate_v_parameter()

    def _calculate_v_parameter(self):
        Consts.print_info("minimize", "Computing v_parameter")
        # For seeing the whole process of LBFGS add '"disp": True' to the 'options' dict
        t1 = time()
        optimize_result = minimize(fun=self._L, x0=np.zeros(self.feature.features_amount),
                                   jac=self._gradient, method="L-BFGS-B", options={"maxiter": 400})
        Consts.TIME = 1
        Consts.print_time("_calculate_v_parameter", time() - t1)
        return optimize_result.x

    def _calculate_matrices(self, v_parameter):
        v_sum = (self.feature.features_matrix_all_possible_histories.dot(v_parameter)).reshape(
            (self.feature.len_tagged_histories, Consts.TAGS_AMOUNT))
        logsumexp_v = logsumexp(v_sum, axis=-1, keepdims=True)
        self.right_sum_for_L = np.sum(logsumexp_v)

        product = np.exp(v_sum - logsumexp_v).reshape((self.feature.len_all_possible_tagged_histories, 1))
        self.product_for_gradient = np.sum(self.feature.features_matrix_all_possible_histories.multiply(product), axis=0)

    def _v_squares(self, v_parameter):
        # Consts.print_debug("_v_squares", "Calculating")
        return np.sum(np.square(v_parameter))

    def _L(self, v_parameter):
        # Consts.print_info("_L", "Calculating")
        self.time_started_LBFGS = time()
        self._calculate_matrices(v_parameter)

        left_sum = np.sum(self.feature.features_matrix_tagged_histories.dot(v_parameter))
        right_sum = self.right_sum_for_L

        return -(left_sum - right_sum - (self._v_squares(v_parameter) * (self.lambda_value / 2)))

    def _gradient(self, v_parameter):
        gradient = self.features_occurrences_ndarray - np.squeeze(np.asarray(self.product_for_gradient)) - \
                   np.multiply(v_parameter, self.lambda_value)

        Consts.TIME = 1
        Consts.print_time("Iterate number " + str(self.iterate_number), time() - self.time_started_LBFGS)
        self.iterate_number += 1

        return gradient*(-1)
