from features import Feature
from consts import Consts
from training import Training

import pickle


class AdvancedModel(object):
    feature = None

    v_parameter = None

    def __init__(self, method: str, file_full_name: str=None):
        if method == Consts.TRAIN:
            self._training(file_full_name)
        elif method == Consts.TAG:
            self._set_internal_values()

    def _training(self, file_full_name: str):
        # TODO: should run with several lambda values
        self.v_parameter = Training(Consts.ADVANCED_MODEL, ["100", "101", "102", "103", "104", "105"], lambda_value=0.1,
                                    file_full_name=file_full_name).v_parameter
        with open("../data_from_training/advanced_model/v_parameter", 'wb') as f:
            pickle.dump(self.v_parameter, f, protocol=-1)

    def _set_internal_values(self):
        with open("../data_from_training/advanced_model/v_parameter", 'rb') as f:
            self.v_parameter = pickle.load(f)
        self.feature = Feature(Consts.TAG, Consts.ADVANCED_MODEL)
