import unittest

from consts import Consts
from basicModel import BasicModel
from features import Feature
from history import History

import numpy as np
from math import exp


class MyTestCase(unittest.TestCase):
    # TODO: before running this test save v_parameter aside because it override it
    def test_calculate_v_for_two_sentences(self):
        basic_model = BasicModel(Consts.TRAIN, "../tests/trialDataFiles/trialTwoSentences.wtag")
        expected_v = [1.1151084504245419, 0.5586696093807878, 0.0, 0.22232446324656183, 0.0, 0.30644326553733625,
                      0.27129997222861285, 0.35404960985869155, 0.35907974762994865, 0.30025653143932868,
                      0.29582462623902001, 0.70024576781845271, 0.3311265939155516, 0.0, 0.2505143080069987,
                      0.33313357468543003, 0.0, 0.20052944124390321, 0.3311265939155516, 0.33033614730098992,
                      0.33388396165179252, 0.30644326553733625, 0.30025653143932868, 0.68864865084301441,
                      0.19881029181724624, 0.27129997222861285, 0.22232446324656183, 0.30644326553733625,
                      0.20052944124390321, 0.23851859632650349, 0.35907974762994865, 0.37979926044029449, 0.0,
                      0.37979926044029449, 0.27129997222861285, 0.19881029181724624, 0.33088675624490488,
                      0.35907974762994865, 0.0, 0.2505143080069987, -0.054139903648458033, 0.2793348046903939, 0.0,
                      1.7644359934818284e-17, 0.0, 0.30644326553733625, 0.0, 0.27129997222861285, 0.17133413415285764,
                      0.16227393620704458, 0.30025653143932868, 0.1574785296254034, 0.19594745726319387,
                      -0.054139903648458033, 0.3311265939155516, 0.0, 0.5010286160139974, 0.2762659635461146, 0.0,
                      0.20052944124390321, 0.3311265939155516, 0.33033614730098992, 0.33388396165179252,
                      0.30644326553733625, 0.30025653143932868, 0.21261009819655544, 0.054139903648458068,
                      0.25988838406389686, 0.27129997222861285, 0.21539187645602659, 0.2793348046903939,
                      1.7644359934818284e-17, 0.30644326553733625, 0.20052944124390321, 0.34601559196493192,
                      0.16227393620704458, 0.37979926044029449, 0.0, 0.37979926044029449, 0.22942751175599507,
                      0.27129997222861285, 0.20036852318901385, 0.054139903648458068, 0.33088675624490488,
                      0.16227393620704458, 0.0, 0.24709913180778537, -0.054139903648458033, 0.036793228681011404, 0.0,
                      1.7644359934818284e-17, 0.0, 0.020236751511728164, 0.0, 0.29234981286438538, 0.39667717848619582,
                      0.012527587395936204, 0.084485893393084946, 0.26297566821128676, 0.12557358420907502,
                      -0.054139903648458033, -0.16457152077269852, 0.0, -0.10481358351115538, 0.0, 0.0,
                      -0.16457152077269852, -0.15856476929788438, 0.016671974390077948, 0.020236751511728164,
                      -0.18889679230648726, 0.054139903648458068, -0.054254736577886577, -0.137399961910009,
                      1.7644359934818284e-17, 0.020236751511728164, 0.0, 0.012527587395936225, -0.08444164912825726,
                      0.0, -0.08444164912825726, -0.060710254535184437, 0.054139903648458068, 0.012527587395936225,
                      -0.11115391901759225]
        self.assertEqual(basic_model.v_parameter.all(), np.asarray(expected_v).all())

    def test_calculate_v_for_all_sentences(self):
        basic_model = BasicModel(Consts.TRAIN, "../" + Consts.PATH_TO_TRAINING)
        with open('../data_from_training/basic_model/v_as_list', 'w+') as f:
            print(list(basic_model.v_parameter), file=f)

    def test_save_internal_fields(self):
        feature = Feature(Consts.TRAIN, Consts.BASIC_MODEL, ["100", "103", "104"], "../" + Consts.PATH_TO_TRAINING)
        basic_model = BasicModel(Consts.TAG)
        self.assertEqual(basic_model.feature.feature_vector, feature.feature_vector)

    def test_log_probability(self):
        basic_model = BasicModel(Consts.TAG)
        history = History(['*', '*'],
                          ['The', 'Treasury', 'is', 'still', 'working', 'out', 'the', 'details', 'with', 'bank',
                           'trade', 'associations', 'and', 'the', 'other', 'government', 'agencies', 'that', 'have',
                           'a', 'hand', 'in', 'fighting', 'money', 'laundering', '.'], 0)
        # self.assertEqual(exp(basic_model.log_probability(history, 24)), 0.9981080577536964)  # tag = 'DT'
        # self.assertEqual(exp(basic_model.log_probability(history, 0)), 3.965426009950786e-05)  # tag = 'VBN'
        self.assertEqual(exp(basic_model.log_probability(history, 24)), 0.9981080556251704)  # tag = 'DT'
        self.assertEqual(exp(basic_model.log_probability(history, 0)), 3.965545161831752e-05)  # tag = 'VBN'

        max_prob = 0
        best_tag = ''
        for tag_idx, tag in enumerate(Consts.POS_TAGS):
            cur_prob = exp(basic_model.log_probability(history, tag_idx))
            if cur_prob > max_prob:
                max_prob = cur_prob
                best_tag = tag
        self.assertEqual(best_tag, 'DT')


if __name__ == '__main__':
    unittest.main()
