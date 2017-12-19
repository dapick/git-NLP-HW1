import unittest

from consts import Consts
from advancedModel import AdvancedModel
from features import Feature


class MyAdvancedModelTestCase(unittest.TestCase):
    # TODO: before running this test save 'v_parameter' and 'internal_values_of_feature' aside because it override it
    def test_calculate_v_for_two_sentences(self):
        advanced_model = AdvancedModel(Consts.TRAIN, "../tests/trialDataFiles/trialTwoSentences.wtag")

    def test_calculate_v_for_all_sentences(self):
        advanced_model = AdvancedModel(Consts.TRAIN, "../" + Consts.PATH_TO_TRAINING)
        with open('../data_from_training/advanced_model/v_as_list', 'w+') as f:
            print(list(advanced_model.v_parameter), file=f)

    def test_save_internal_fields(self):
        feature = Feature(Consts.TRAIN, Consts.BASIC_MODEL, ["100", "103", "104"], "../" + Consts.PATH_TO_TRAINING)
        advanced_model = AdvancedModel(Consts.TAG)
        self.assertEqual(advanced_model.feature.feature_vector, feature.feature_vector)


if __name__ == '__main__':
    unittest.main()
