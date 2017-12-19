import unittest

from basicModel import BasicModel
from consts import Consts
from tagger import Tagger


class TestTrainAndTag(unittest.TestCase):
    def test_basic_test_training_and_tagging(self):
        basic_model = BasicModel(Consts.TRAIN, "../" + Consts.PATH_TO_TRAINING)
        with open('../data_from_training/basic_model/v_as_list', 'w+') as f:
            for derivative in basic_model.v_parameter:
                print(derivative, file=f)
        file_tagger = Tagger("../data/test.words")
        file_tagger.tag()
        file_tagger.calculate_accuracy("../data/test.wtag", "../data/test_output.wtag")


if __name__ == '__main__':
    unittest.main()
