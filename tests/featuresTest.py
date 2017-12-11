import unittest

from features import Feature
from consts import Consts


class FeaturesTestCase(unittest.TestCase):
    def test_count_features_types(self):
        feature = Feature("../" + Consts.PATH_TO_TRAINING)
        feature.count_features_types()


if __name__ == '__main__':
    unittest.main()
