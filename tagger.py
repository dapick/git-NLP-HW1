from consts import Consts
from viterbi import Viterbi
from parsing import Parsing
from basicModel import BasicModel
from history import Histories

from multiprocessing.pool import Pool
from time import time


class Tagger:
    # input file
    word_file = None
    sentences_list = None

    # output file
    tagged_file = None
    tags_list = None

    # tagger
    viterbi = None

    basic_model = None

    def __init__(self, word_file: str):
        self.word_file = word_file
        self.sentences_list = Parsing.parse_words_file_to_list(self.word_file)
        self.basic_model = BasicModel(Consts.TAG)
        self.tags_per_word = Histories.build_history_tag_per_word_lists("../data/train.wtag")

        self.tagged_file = self._get_tagged_file_name()
        self.tags_list = []

    def _get_tagged_file_name(self):
        ret_file = self.word_file
        ret_file = ret_file.replace("words", "wtag")
        ret_file = ret_file.replace("/", "/output_")
        return ret_file

    def _tag_sentence(self, sentence_tuple: tuple):
        # Consts.TIME = 1
        sentence_idx, sentence = sentence_tuple
        # t1 = time()
        self.viterbi = Viterbi(sentence, self.basic_model)
        tags = self.viterbi.run_viterbi()
        # TODO: add 1 to the sentence index
        # Consts.print_time("Tagging sentence " + str(sentence_idx), time() - t1)
        return tags

    def tag(self):
        # Consts.print_info("tag_file", "Tagging file " + self.word_file)

        # Run parallel - good when checking many sentences
        with Pool(6) as pool:
            sentences_tags = pool.map(self._tag_sentence, enumerate(self.sentences_list))

        # Run linear - good when checking a few sentences
        # for sen in self.sentences_list:
        #     self._tag_sentence(sen)

        Parsing.parse_lists_to_wtag_file(self.sentences_list, list(sentences_tags), self.tagged_file)

    @staticmethod
    def calculate_accuracy(out_file: str, expected_file: str):
        out_list_w, out_list_t = Parsing.parse_wtag_file_to_lists(out_file)
        exp_list_w, exp_list_t = Parsing.parse_wtag_file_to_lists(expected_file)

        count = 0
        for i, sen in enumerate(out_list_w):
            for j, w in enumerate(sen):
                if w == exp_list_w[i][j] and out_list_t[i][j] == exp_list_t[i][j]:
                    count += 1
        num_words = sum(len(out_list_w[k]) for k in range(0, len(out_list_w)))
        str_res = str(100 * count / num_words) + "%"
        print("The accuracy is: " + str_res)
