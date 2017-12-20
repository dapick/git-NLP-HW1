import unittest

from basicModel import BasicModel
from advancedModel import AdvancedModel
from consts import Consts
from tagger import Tagger


class TestTrainAndTag(unittest.TestCase):
    def test_basic_model_training_and_tagging(self):
        # basic_model = BasicModel(Consts.TRAIN, "../" + Consts.PATH_TO_TRAINING)
        # with open('../data_from_training/basic_model/v_as_list', 'w+') as f:
        #     for derivative in basic_model.v_parameter:
        #         print(derivative, file=f)
        test_words_path = "../data/test.words"
        file_tagger = Tagger(test_words_path)
        # file_tagger.tag()
        file_tagger.calculate_accuracy("../data/output_test.wtag", "../data/test.wtag")

    def test_advanced_model_training_and_tagging(self):
        advanced_model = AdvancedModel(Consts.TRAIN, "../" + Consts.PATH_TO_TRAINING)
        with open('../data_from_training/advanced_model/v_as_list', 'w+') as f:
            for derivative in advanced_model.v_parameter:
                print(derivative, file=f)
        test_words_path = "../data/test.words"
        file_tagger = Tagger(test_words_path)
        file_tagger.tag()
        file_tagger.calculate_accuracy(file_tagger.tagged_file, "../data/test.wtag")


if __name__ == '__main__':
    unittest.main()
