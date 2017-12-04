import unittest
from history import History, Histories


class HistoryTestCase(unittest.TestCase):
    def test_prefix(self):
        self.history = History("Hello")
        self.assertTrue(self.history.starts_with("H"))
        self.assertTrue(self.history.starts_with("He"))
        self.assertTrue(self.history.starts_with("Hell"))
        self.assertFalse(self.history.starts_with("e"))
        self.assertFalse(self.history.starts_with("ell"))

    def test_suffix(self):
        self.history = History("Hello")
        self.assertTrue(self.history.ends_with("o"))
        self.assertTrue(self.history.ends_with("lo"))
        self.assertTrue(self.history.ends_with("ello"))
        self.assertFalse(self.history.ends_with("l"))
        self.assertFalse(self.history.ends_with("ell"))

    def test_build_history_list_from_one_sentence(self):
        history_one_sentences_file_name = "trialDataFiles/trialOneSentence.wtag"
        # TODO: quite hard to check it. Looks good for now - 04/12
        histories = Histories.build_history_list(history_one_sentences_file_name)
        print("Finished")

    def test_build_history_list_from_two_sentence(self):
        history_one_sentences_file_name = "trialDataFiles/trialTwoSentences.wtag"
        # TODO: quite hard to check it. Looks good for now - 04/12
        histories = Histories.build_history_list(history_one_sentences_file_name)
        print("Finished")


if __name__ == '__main__':
    unittest.main()
