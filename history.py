from parsing import Parsing
from consts import Consts


class History(object):
    tags = None
    sentence = None
    current_word_idx = None

    # Creates a new History instance
    def __init__(self, tags: list, sentence: list, current_word_idx: int):
        self.tags = tags
        self.sentence = sentence
        self.current_word_idx = current_word_idx

    def get_current_word(self):
        return self.sentence[self.current_word_idx]


class Histories(object):
    # Returns a list of all possible histories and the tags given
    @staticmethod
    def build_history_list_and_tags_list(file_full_name: str=Consts.PATH_TO_TRAINING) -> (list, list):
        sentences, tags = Parsing.parse_wtag_file_to_lists(file_full_name)
        histories = []
        histories_tags = []
        for sentence, sentence_tags in zip(sentences, tags):
            for idx, tag in enumerate(sentence_tags):
                if idx > 1:
                    histories.append(
                        History([sentence_tags[idx-2], sentence_tags[idx-1]], sentence, idx))
                else:
                    if idx == 0:
                        histories.append(
                            History(["*", "*"], sentence, idx))
                    else:
                        histories.append(
                            History(["*", sentence_tags[idx-1]], sentence, idx))
                histories_tags.append(tag)
        return histories, histories_tags
