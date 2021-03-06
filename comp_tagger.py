from consts import Consts
from tagger import Tagger


class CompTagger(object):

    def __init__(self, model: str):
        if model == Consts.BASIC_MODEL:
            self.model = Consts.BASIC_MODEL
        elif model == Consts.ADVANCED_MODEL:
            self.model = Consts.ADVANCED_MODEL

    def tag_comp(self):
        file_tagger = Tagger("data/comp.words", self.model)
        file_tagger.tag()


if __name__ == '__main__':
    basic_comp_tagger = CompTagger(Consts.BASIC_MODEL)
    basic_comp_tagger.tag_comp()
    advanced_comp_tagger = CompTagger(Consts.ADVANCED_MODEL)
    advanced_comp_tagger.tag_comp()

    # TODO: not for the final version
    basic_tagger = Tagger("../tests/trialDataFiles/trialTwoSentences.words", Consts.BASIC_MODEL)
    basic_tagger.calculate_accuracy(basic_tagger.tagged_file, "../tests/trialDataFiles/trialTwoSentences.wtag")
    advanced_tagger = Tagger("../tests/trialDataFiles/trialTwoSentences.words", Consts.ADVANCED_MODEL)
    advanced_tagger.calculate_accuracy(advanced_tagger.tagged_file, "../tests/trialDataFiles/trialTwoSentences.wtag")