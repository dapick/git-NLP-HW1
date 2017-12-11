import unittest

from consts import Consts
from basicModel import BasicModel


class MyTestCase(unittest.TestCase):
    def test_v_parameter_calculation(self):
        basic_model = BasicModel("../" + Consts.PATH_TO_TRAINING)
        basic_model.calculate_v_parameter()
        print("The final v is:", basic_model.v_parameter)


if __name__ == '__main__':
    unittest.main()
