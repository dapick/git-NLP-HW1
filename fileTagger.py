import numpy as np
from consts import Consts
from viterbi import Viterbi
from parsing import Parsing
from multiprocessing import Pool


class Tagger:

    # input file
    word_file = None
    sentences_list = None

    # output file
    tagged_file = None
    tags_list = None

    # tagger
    viterbi = None

    def __init__(self, word_file: str=Consts.PATH_TO_COMPETITION):
        self.word_file = word_file
        self.sentences_list = Parsing.parse_words_file_to_list(self.word_file)

        self.tagged_file = self._get_tagged_file_name()
        self.tags_list = []

    def _get_tagged_file_name(self):
        ret_file = self.word_file
        return ret_file.replace("words", "wtag")

    def _tag_sentence(self, sen: list) ->list:
        Consts.print_info("sentence_tagger", "Tagging sentence " + str(sen))
        self.viterbi = Viterbi(sen)

        tags = self.viterbi.run_viterbi()
        self.tags_list.append(tags)

    def tag(self):
        Consts.print_info("tag_file", "Tagging file " + self.word_file)
        pool = Pool(4)
        pool.map(self._tag_sentence, self.sentences_list)
        #for sen in self.sentences_list:
         #   self._tag_sentence(sen)

        Parsing.parse_lists_to_wtag_file(self.sentences_list, self.tags_list, self.tagged_file)

    # TODO: not done
    @staticmethod
    def calculate_percision(out_file: str, expected_file: str):
        out_list_w, out_list_t = Parsing.parse_wtag_file_to_lists(out_file)
        exp_list_w, exp_list_t = Parsing.parse_wtag_file_to_lists(expected_file)

        count = 0
        for i, sen in enumerate(out_list_w):
            for j, w in enumerate(sen):
                if w == exp_list_w[i][j] and out_list_t[i][j] == exp_list_t[i][j]:
                    count += 1
        num_words = sum(len(out_list_w[k]) for k in range(0, len(out_list_w)))
        print(num_words)
        str_res = str(100 * count / num_words)+"%"
        return str_res


if __name__ == '__main__':
    file_tagger = Tagger("tests/trialDataFiles/short.words")
    file_tagger.tag()
    # print(file_tagger.calculate_percision("tests/trialDataFiles/short.wtag", "tests/trialDataFiles/short.exp"))