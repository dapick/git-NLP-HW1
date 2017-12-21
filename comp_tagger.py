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
    tag = CompTagger("<MODEL_TYPE>_model")
    tag.tag_comp()
