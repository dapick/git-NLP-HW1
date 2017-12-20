from consts import Consts
from features import Feature

from scipy.optimize import minimize
import numpy as np
from time import time
from scipy.sparse import coo_matrix


class Training(object):
    feature = None
    features_amount = None
    features_occurrences_ndarray = None

    v_parameter = None
    # TODO: find which lambda should we use
    lambda_value = None

    # Common values between iterations
    right_sum_for_L = None
    product_for_gradient = None

    def __init__(self, model: str, used_features: list, lambda_value: float, file_full_name: str=Consts.PATH_TO_TRAINING):
        self.feature = Feature(Consts.TRAIN, model, used_features, file_full_name)
        self.features_amount = len(self.feature.features_occurrences)
        self.features_occurrences_ndarray = np.asarray(self.feature.features_occurrences)
        self.lambda_value = np.array(lambda_value)

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

    def _calculate_matrices(self, v_parameter):
        data = []
        rows = []
        cols = []
        for history_idx, tagged_history in enumerate(self.feature.all_possible_tagged_histories):
            list_idxs = self.feature.history_tag_features[tagged_history]
            len_list_idx = len(list_idxs)
            data += [1] * len_list_idx
            rows += [history_idx] * len_list_idx
            cols += list_idxs
        features_matrix = coo_matrix((data, (rows, cols)),
                                     shape=(self.feature.len_all_possible_tagged_histories, self.features_amount)).tocsr()

        numerator_for_gradient = np.exp(features_matrix.dot(v_parameter)).reshape(
            (self.feature.len_histories_in_train, Consts.TAGS_AMOUNT))
        inner_sum = np.sum(numerator_for_gradient, axis=-1).reshape((self.feature.len_histories_in_train, 1))
        self.right_sum_for_L = np.sum(np.log(inner_sum))

        product = (numerator_for_gradient / inner_sum).reshape(
            (self.feature.len_all_possible_tagged_histories, 1))
        self.product_for_gradient = np.sum(features_matrix.multiply(product), axis=0)

    def _v_squares(self, v_parameter):
        # Consts.print_debug("_v_squares", "Calculating")
        return sum(np.square(v_parameter))

    def _L(self, v_parameter):
        # Consts.print_info("_L", "Calculating")
        self.time_started_LBFGS = time()
        self._calculate_matrices(v_parameter)

        left_sum = sum(v_parameter * self.features_occurrences_ndarray)
        right_sum = self.right_sum_for_L

        return -(left_sum - right_sum - ((self.lambda_value/2) * self._v_squares(v_parameter)))

    def _gradient(self, v_parameter):
        gradient = self.features_occurrences_ndarray - self.product_for_gradient - (self.lambda_value * v_parameter)

        Consts.TIME = 1
        Consts.print_time("Iterate number " + str(self.iterate_number), time() - self.time_started_LBFGS)
        self.iterate_number += 1

        return gradient*(-1)
