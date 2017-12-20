from features import Feature
from consts import Consts
from training import Training

import pickle
import abc


class Model(metaclass=abc.ABCMeta):
    feature = None
    v_parameter = None

    def __init__(self, method: str, file_full_name: str=None):
        if method == Consts.TRAIN:
            self._training(file_full_name)
        elif method == Consts.TAG:
            self._set_internal_values()

    @abc.abstractmethod
    def _training(self, file_full_name: str):
        raise NotImplementedError

    @abc.abstractmethod
    def _set_internal_values(self):
        raise NotImplementedError


class BasicModel(Model):
    def __init__(self, method: str, file_full_name: str=Consts.PATH_TO_TRAINING):
        super().__init__(method, file_full_name)

    def _training(self, file_full_name: str):
        self.v_parameter = Training(Consts.BASIC_MODEL, ["100", "103", "104"], lambda_value=0.1,
                                    file_full_name=file_full_name).v_parameter
        with open("../data_from_training/" + Consts.BASIC_MODEL + "/v_parameter", 'wb') as f:
            pickle.dump(self.v_parameter, f, protocol=-1)

    def _set_internal_values(self):
        with open("../data_from_training/" + Consts.BASIC_MODEL + "/v_parameter", 'rb') as f:
            self.v_parameter = pickle.load(f)
        self.feature = Feature(Consts.TAG, Consts.BASIC_MODEL)


class AdvancedModel(Model):
    def __init__(self, method: str, file_full_name: str=None):
        super().__init__(method, file_full_name)

    def _training(self, file_full_name: str):
        # TODO: should run with several lambda values
        self.v_parameter = Training(Consts.ADVANCED_MODEL,
                                    ["100", "101", "102", "103", "104", "105", "capital_letter", "hyphen_word"],
                                    lambda_value=0.1, file_full_name=file_full_name).v_parameter
        with open("../data_from_training/" + Consts.ADVANCED_MODEL + "/v_parameter", 'wb') as f:
            pickle.dump(self.v_parameter, f, protocol=-1)

    def _set_internal_values(self):
        with open("../data_from_training/" + Consts.ADVANCED_MODEL + "/v_parameter", 'rb') as f:
            self.v_parameter = pickle.load(f)
        self.feature = Feature(Consts.TAG, Consts.ADVANCED_MODEL)
