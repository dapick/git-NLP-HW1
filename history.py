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

    def get_current_word(self) -> str:
        return self.sentence[self.current_word_idx]

    # Returns the current word idx first letters and lower case
    def word_custom_prefix(self, idx: int) -> str:
        return self.sentence[self.current_word_idx][:idx].lower()

    # Returns the current word idx last letters and lower case
    def word_custom_suffix(self, idx: int) -> str:
        return self.sentence[self.current_word_idx][-idx:].lower()


class TaggedHistory(History):
    tag_idx = None

    def __init__(self, tags: list, sentence: list, current_word_idx: int, tag_idx: int):
        super().__init__(tags, sentence, current_word_idx)
        self.tag_idx = tag_idx

    def get_tag_name(self):
        return Consts.POS_TAGS[self.tag_idx]


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

    # Returns a list of all possible histories and the tags given
    @staticmethod
    def build_tagged_histories_list(file_full_name: str) -> list:
        sentences, tags = Parsing.parse_wtag_file_to_lists(file_full_name)
        tagged_histories = []
        for sentence, sentence_tags in zip(sentences, tags):
            for idx_in_sentence, tag in enumerate(sentence_tags):
                tag_idx = Consts.DICT_POS_TAGS[tag]
                if idx_in_sentence > 1:
                    tagged_histories.append(
                        TaggedHistory([sentence_tags[idx_in_sentence-2], sentence_tags[idx_in_sentence-1]], sentence, idx_in_sentence, tag_idx))
                else:
                    if idx_in_sentence == 0:
                        tagged_histories.append(
                            TaggedHistory(["*", "*"], sentence, idx_in_sentence, tag_idx))
                    elif idx_in_sentence == 1:
                        tagged_histories.append(
                            TaggedHistory(["*", sentence_tags[idx_in_sentence-1]], sentence, idx_in_sentence, tag_idx))
        return tagged_histories
