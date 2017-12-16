from consts import Consts
from viterbi import Viterbi
from parsing import Parsing
from basicModel import BasicModel


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

    def __init__(self, word_file: str=Consts.PATH_TO_COMPETITION):
        self.word_file = word_file
        self.sentences_list = Parsing.parse_words_file_to_list(self.word_file)
        self.basic_model = BasicModel(Consts.TAG)

        self.tagged_file = self._get_tagged_file_name()
        self.tags_list = []

    def _get_tagged_file_name(self):
        ret_file = self.word_file
        return ret_file.replace("words", "wtag")

    def _tag_sentence(self, sen: list):
        Consts.print_info("sentence_tagger", "Tagging sentence " + str(sen))
        self.viterbi = Viterbi(sen, self.basic_model)
        tags = self.viterbi.run_viterbi()
        self.tags_list.append(tags)

    def tag(self):
        Consts.print_info("tag_file", "Tagging file " + self.word_file)
        for sen in self.sentences_list:
            self._tag_sentence(sen)

        Parsing.parse_lists_to_wtag_file(self.sentences_list, self.tags_list, self.tagged_file)

    # TODO: not done
    def calculate_percision(self, compare_file: str):
        compare_wtag_list = Parsing.parse_wtag_file_to_lists(compare_file)
