import unittest

from tagger import Tagger


class TaggerTestCase(unittest.TestCase):
    def test_short_sentence(self):
        file_tagger = Tagger("trialDataFiles/short.words")
        file_tagger.tag()


if __name__ == '__main__':
    unittest.main()
