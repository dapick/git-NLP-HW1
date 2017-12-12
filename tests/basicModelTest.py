import unittest

from consts import Consts
from basicModel import BasicModel
from fileTagger import Tagger


class MyTestCase(unittest.TestCase):
    def test_v_parameter_calculation(self):
        # basic_model = BasicModel("../" + Consts.PATH_TO_TRAINING)
        basic_model = BasicModel("trialDataFiles/trialOneSentence.wtag")
        basic_model.calculate_v_parameter()
        print("The final v is:", basic_model.v_parameter)
        tagger = Tagger("trialDataFiles/trialOneSentence.words")
        tagger.tag_file()


if __name__ == '__main__':
    unittest.main()
