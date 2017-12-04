import unittest

from history import History
from features import Feature


class FeaturesTestCase(unittest.TestCase):
    def test_feature_100(self):
        history = History("base")
        self.assertTrue(Feature.feature_100(history, "Vt"))
        self.assertFalse(Feature.feature_100(history, "JJ"))
        history = History("basketball")
        self.assertFalse(Feature.feature_100(history, "Vt"))

    def test_feature_101(self):
        history = History("cycling")
        self.assertTrue(Feature.feature_101(history, "VBG"))
        history = History("ing")
        self.assertTrue(Feature.feature_101(history, "VBG"))
        self.assertFalse(Feature.feature_101(history, "JJ"))
        history = History("ingredients")
        self.assertFalse(Feature.feature_101(history, "VBG"))

    def test_feature_102(self):
        history = History("Preparation")
        self.assertTrue(Feature.feature_102(history, "NN"))
        history = History("Pre")
        self.assertTrue(Feature.feature_102(history, "NN"))
        self.assertFalse(Feature.feature_102(history, "VBG"))
        history = History("abpre")
        self.assertFalse(Feature.feature_102(history, "NN"))

    def test_feature_103(self):
        history = History("bla", ("DT", "JJ"))
        self.assertTrue(Feature.feature_103(history, "Vt"))
        self.assertFalse(Feature.feature_103(history, "NN"))
        history = History("bla", ("DT", "Vt"))
        self.assertFalse(Feature.feature_103(history, "Vt"))
        history = History("bla", ("Vt", "JJ"))
        self.assertFalse(Feature.feature_103(history, "Vt"))

    def test_feature_104(self):
        history = History("bla", ("*", "JJ"))
        self.assertTrue(Feature.feature_104(history, "Vt"))
        self.assertFalse(Feature.feature_104(history, "NN"))
        history = History("bla", ("JJ", "Vt"))
        self.assertFalse(Feature.feature_104(history, "Vt"))

    def test_feature_105(self):
        history = History("bla")
        self.assertTrue(Feature.feature_105(history, "Vt"))
        history = History("bla", ("JJ", "Vt"))
        self.assertTrue(Feature.feature_105(history, "Vt"))
        self.assertFalse(Feature.feature_105(history, "NN"))


if __name__ == '__main__':
    unittest.main()
