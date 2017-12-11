import numpy as np
from consts import Consts
from viterbi import Viterbi
from parsing import Parsing


class Tagger:

    # input file
    word_file = None
    word_dict = None

    # output file
    tagged_file = None
    word_tagged_dict = None

    # tagger
    viterbi = None

    def __init__(self):
        # self.word_file = Consts.PATH_TO_COMPETITION
        self.word_file = "tests/trialDataFiles/trialTwoSentences.words"
        self.word_dict = Parsing().parse_words_file_to_list(self.word_file)

        self.word_tagged_dict = {}

    def sentence_tagger(self, sen: list) ->list:
        self.viterbi = Viterbi(sen)
        self.word_tagged_dict[0] = self.viterbi.run_viterbi()

    def print_to_file(self):
        with open(self.tagged_file, 'r+') as f:
            for sen in sorted(self.word_tagged_dict.iterkeys()):
                [print(word_tag, file=f) for word_tag in sen]





if __name__ == '__main__':
    tagger = Tagger()
    print(tagger.word_dict)
    tagger.sentence_tagger(tagger.word_dict[0])
    print(tagger.word_tagged_dict[0])


