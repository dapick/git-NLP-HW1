import unittest

from parsing import Parsing
from consts import Consts


class ParseWTagsTestCase(unittest.TestCase):
    def test_parse_wtag_file_to_lists_with_one_sentence(self):
        one_sentence_file_name = "trialDataFiles/trialOneSentence.wtag"
        sentences, tags = Parsing.parse_wtag_file_to_lists(one_sentence_file_name)
        expected_parsing = [("The", "DT"), ("Treasury", "NNP"), ("is", "VBZ"), ("still", "RB"), ("working", "VBG"), ("out", "RP"), ("the", "DT"), ("details", "NNS"), ("with", "IN"), ("bank", "NN"), ("trade", "NN"), ("associations", "NNS"), ("and", "CC"), ("the", "DT"), ("other", "JJ"), ("government", "NN"), ("agencies", "NNS"), ("that", "WDT"), ("have", "VBP"), ("a", "DT"), ("hand", "NN"), ("in", "IN"), ("fighting", "VBG"), ("money", "NN"), ("laundering", "NN"), (".", ".")]
        for word, tag, item in zip(sentences[0], tags[0], expected_parsing):
            self.assertEqual(word, item[0])
            self.assertEqual(tag, item[1])

    def test_parse_wtag_file_to_lists_with_two_sentences(self):
        two_sentences_file_name = "trialDataFiles/trialTwoSentences.wtag"
        sentences, tags = Parsing.parse_wtag_file_to_lists(two_sentences_file_name)
        # expected_parsing = [[("The", "DT"), ("Treasury", "NNP"), ("is", "VBZ"), ("still", "RB"), ("working", "VBG"), ("out", "RP"), ("the", "DT"), ("details", "NNS"), ("with", "IN"), ("bank", "NN"), ("trade", "NN"), ("associations", "NNS"), ("and", "CC"), ("the", "DT"), ("other", "JJ"), ("government", "NN"), ("agencies", "NNS"), ("that", "WDT"), ("have", "VBP"), ("a", "DT"), ("hand", "NN"), ("in", "IN"), ("fighting", "VBG"), ("money", "NN"), ("laundering", "NN"), (".", ".")],
        #                     [("Among", "IN"), ("the", "DT"), ("possibilities", "NNS"), ("the", "DT"), ("Treasury", "NNP"), ("is", "VBZ"), ("considering", "VBG"), ("are", "VBP"), ("requirements", "NNS"), ("that", "IN"), ("banks", "NNS"), ("keep", "VB"), ("records", "NNS"), ("identifying", "VBG"), ("the", "DT"), ("originators", "NNS"), ("and", "CC"), ("recipients", "NNS"), ("of", "IN"), ("international", "JJ"), ("wire", "NN"), ("transfers", "NNS"), (".", ".")]]
        # for sentence, sentence_tags, expected_sentence in zip(sentences, tags, expected_parsing):
        #     for word, tag, item in zip(sentence, sentence_tags, expected_sentence):
        #         self.assertEqual(word, item[0])
        #         self.assertEqual(tag, item[1])
        with open("trialDataFiles/short.words", 'w') as f:
            for sentence in sentences:

                print(sentence, file=f)

    def test_parse_words_file_to_list_with_one_sentence(self):
        one_sentence_file_name = "trialDataFiles/trialOneSentence.words"
        sentences = Parsing.parse_words_file_to_list(one_sentence_file_name)
        expected_parsing = ["Traders", "said", "there", "were", "some", "busy", "dealings", "in", "Freddie", "Mac", "and", "Federal", "National", "Mortgage", "Association", "securities", "because", "underwriters", "from", "last", "week", "'s", "heavy", "slate", "of", "real", "estate", "mortgage", "investment", "conduit", "issues", "moved", "to", "gather", "collateral", "for", "new", "deals", "."]

        for word, expected_word in zip(sentences[0], expected_parsing):
            self.assertEqual(word, expected_word)

    def test_parse_words_file_to_list_with_two_sentences(self):
        two_sentences_file_name = "trialDataFiles/trialTwoSentences.words"
        sentences = Parsing.parse_words_file_to_list(two_sentences_file_name)
        expected_parsing = [["Traders", "said", "there", "were", "some", "busy", "dealings", "in", "Freddie", "Mac", "and", "Federal", "National", "Mortgage", "Association", "securities", "because", "underwriters", "from", "last", "week", "'s", "heavy", "slate", "of", "real", "estate", "mortgage", "investment", "conduit", "issues", "moved", "to", "gather", "collateral", "for", "new", "deals", "."],
                            ["The", "spokesman", "further", "said", "that", "at", "least", "two", "more", "offers", "are", "expected", "from", "other", "companies", "within", "two", "weeks", "."]]

        for sentence, expected_sentence in zip(sentences, expected_parsing):
            for word, expected_word in zip(sentence, expected_sentence):
                self.assertEqual(word, expected_word)

    def test_parse_lists_to_wtag_file_with_one_sentence(self):
        one_sentence_file_name = "trialDataFiles/trialOneSentence.wtag"
        sentences, tags = Parsing.parse_wtag_file_to_lists(one_sentence_file_name)
        output_one_sentence_file_name = "trialDataFiles/outputTrialOneSentence.wtag"
        Parsing.parse_lists_to_wtag_file(sentences, tags, output_one_sentence_file_name)

        with open(output_one_sentence_file_name) as f:
            file1_data = f.read()
        with open(one_sentence_file_name) as f:
            file2_data = f.read()
        self.assertEqual(file1_data, file2_data)

    def test_parse_lists_to_wtag_file_with_two_sentences(self):
        two_sentences_file_name = "trialDataFiles/trialTwoSentences.wtag"
        sentences, tags = Parsing.parse_wtag_file_to_lists(two_sentences_file_name)
        output_two_sentences_file_name = "trialDataFiles/outputTrialTwoSentences.wtag"
        Parsing.parse_lists_to_wtag_file(sentences, tags, output_two_sentences_file_name)

        with open(output_two_sentences_file_name) as f:
            file1_data = f.read()
        with open(two_sentences_file_name) as f:
            file2_data = f.read()
        self.assertEqual(file1_data, file2_data)

    def test_parse_lists_to_wtag_file_with_all_sentences(self):
        all_sentences_file_name = "../" + Consts.PATH_TO_TRAINING
        sentences, tags = Parsing.parse_wtag_file_to_lists(all_sentences_file_name)
        output_all_sentences_file_name = "trialDataFiles/outputTrialAllSentences.wtag"
        Parsing.parse_lists_to_wtag_file(sentences, tags, output_all_sentences_file_name)

        with open(output_all_sentences_file_name) as f:
            file1_data = f.read()
        with open(all_sentences_file_name) as f:
            file2_data = f.read()
        self.assertEqual(file1_data, file2_data)


if __name__ == '__main__':
    unittest.main()
