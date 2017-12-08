import unittest
from history import History, Histories


class HistoryTestCase(unittest.TestCase):
    def test_get_current_word(self):
        history = History(["*", "*"], ["Hello", "World"], 0)
        self.assertEqual(history.get_current_word(), "Hello")

    def test_prefix(self):
        self.history = History(["*", "*"], ["Hello"], 0)
        self.assertEqual(self.history.word_custom_prefix(1), "H")
        self.assertEqual(self.history.word_custom_prefix(2), "He")
        self.assertEqual(self.history.word_custom_prefix(4), "Hell")

    def test_suffix(self):
        self.history = History(["*", "*"], ["Hello"], 0)
        self.assertEqual(self.history.word_custom_suffix(1), "o")
        self.assertEqual(self.history.word_custom_suffix(2), "lo")
        self.assertEqual(self.history.word_custom_suffix(4), "ello")

    def test_build_history_list_from_one_sentence(self):
        history_one_sentence_file_name = "trialDataFiles/trialOneSentence.wtag"
        histories, tags = Histories.build_history_list_and_tags_list(history_one_sentence_file_name)
        with open("trialDataFiles/outputHistoryTrialOneSentence.output", 'r+') as f:
            for history, sentence_tags in zip(histories, tags):
                print(history.tags, history.sentence, history.current_word_idx, sentence_tags, file=f)
            f.seek(0, 0)
            file1_data = f.read()
        with open("trialDataFiles/expectedHistoryTrialOneSentence.output") as f:
            file2_data = f.read()

        self.assertEqual(file1_data, file2_data)

    def test_build_history_list_from_two_sentence(self):
        history_two_sentences_file_name = "trialDataFiles/trialTwoSentences.wtag"
        histories, tags = Histories.build_history_list_and_tags_list(history_two_sentences_file_name)
        with open("trialDataFiles/outputHistoryTrialTwoSentences.output", 'r+') as f:
            for history, sentence_tags in zip(histories, tags):
                print(history.tags, history.sentence, history.current_word_idx, sentence_tags, file=f)
            f.seek(0, 0)
            file1_data = f.read()
        with open("trialDataFiles/expectedHistoryTrialTwoSentences.output") as f:
            file2_data = f.read()

        self.maxDiff = None
        self.assertEqual(file1_data, file2_data)


if __name__ == '__main__':
    unittest.main()
