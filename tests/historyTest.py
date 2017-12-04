import unittest
from history import History


class HistoryTestCase(unittest.TestCase):
    def setUp(self):
        self.history = History("Hello")

    def test_prefix(self):
        self.assertTrue(self.history.starts_with("H"))
        self.assertTrue(self.history.starts_with("He"))
        self.assertTrue(self.history.starts_with("Hell"))
        self.assertFalse(self.history.starts_with("e"))
        self.assertFalse(self.history.starts_with("ell"))

    def test_suffix(self):
        self.assertTrue(self.history.ends_with("o"))
        self.assertTrue(self.history.ends_with("lo"))
        self.assertTrue(self.history.ends_with("ello"))
        self.assertFalse(self.history.ends_with("l"))
        self.assertFalse(self.history.ends_with("ell"))


if __name__ == '__main__':
    unittest.main()
