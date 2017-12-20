import unittest

from model import BasicModel, AdvancedModel
from consts import Consts
from tagger import Tagger


class TestTrainAndTag(unittest.TestCase):
    def setUp(self):
        self.train_file_path = "../" + Consts.PATH_TO_TRAINING
        self.test_words_path = "../" + Consts.PATH_TO_TEST_WORDS
        self.test_expected_wtag_path = "../" + Consts.PATH_TO_TEST_WTAG

    def test_basic_model_training_and_tagging(self):
        basic_model = BasicModel(Consts.TRAIN, self.train_file_path)
        with open('../data_from_training/' + Consts.BASIC_MODEL + '/v_as_list', 'w+') as f:
            for derivative in basic_model.v_parameter:
                print(derivative, file=f)

        file_tagger = Tagger(self.test_words_path, Consts.BASIC_MODEL)
        file_tagger.tag()
        file_tagger.calculate_accuracy(file_tagger.tagged_file, self.test_expected_wtag_path)

    def test_advanced_model_training_and_tagging(self):
        advanced_model = AdvancedModel(Consts.TRAIN, self.train_file_path)
        with open('../data_from_training/' + Consts.ADVANCED_MODEL + '/v_as_list', 'w+') as f:
            for derivative in advanced_model.v_parameter:
                print(derivative, file=f)

        file_tagger = Tagger(self.test_words_path, Consts.ADVANCED_MODEL)
        file_tagger.tag()
        file_tagger.calculate_accuracy(file_tagger.tagged_file, self.test_expected_wtag_path)


if __name__ == '__main__':
    unittest.main()
