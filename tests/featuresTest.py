import unittest

from features import Feature
from consts import Consts


class FeaturesTestCase(unittest.TestCase):
    def test_count_features_types(self):
        feature = Feature(Consts.TRAIN, "../" + Consts.PATH_TO_TRAINING)
        feature.count_features_types()

    def test_save_internal_fields(self):
        feature = Feature(Consts.TRAIN, ["100", "103", "104"], "../" + Consts.PATH_TO_TRAINING)
        new_feature = Feature(Consts.TAG)
        self.assertEqual(feature.feature_vector.items(), new_feature.feature_vector.items())
        self.assertEqual(feature.used_features, new_feature.used_features)


if __name__ == '__main__':
    unittest.main()
