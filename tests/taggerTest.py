import unittest

from tagger import Tagger


class TaggerTestCase(unittest.TestCase):
    def test_short_sentence(self):
        file_tagger = Tagger("trialDataFiles/test25Sentences.words")
        file_tagger.tag()
        file_tagger.calculate_accuracy(file_tagger.tagged_file, "trialDataFiles/test25Sentences.exep")

    def test_accuracy(self):
        file_tagger = Tagger("trialDataFiles/short.words")
        # file_tagger.tag()
        # file_tagger.calculate_accuracy("trialDataFiles/expectedShort.wtag", file_tagger.tagged_file)


if __name__ == '__main__':
    unittest.main()
