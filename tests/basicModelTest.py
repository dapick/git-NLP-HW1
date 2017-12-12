import unittest

from consts import Consts
from basicModel import BasicModel
from fileTagger import Tagger


class MyTestCase(unittest.TestCase):
    def test_v_parameter_calculation(self):
        basic_model = BasicModel(Consts.TRAIN, "../data/test.wtag")
        print("The final v is:", basic_model.v_parameter)


if __name__ == '__main__':
    unittest.main()
