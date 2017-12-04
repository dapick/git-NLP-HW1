from parsing import Parsing


class History(object):
    current_word = None
    tags = None

    # Creates a new History instance.
    # Saves the current word and the last two tags.
    # If no tags were sent, saves the first two tags as two '*'
    def __init__(self, current_word: str, tags: tuple=("*", "*")):
        self.current_word = current_word
        self.tags = tags

    def __hash__(self):
        return hash((self.current_word, self.tags))

    def __eq__(self, other):
        return (self.current_word, self.tags) == (other.current_word, other.tags)

    def starts_with(self, prefix: str) -> bool:
        return self.current_word.startswith(prefix)

    def ends_with(self, suffix: str) -> bool:
        return self.current_word.endswith(suffix)


class Histories(object):
    # Returns a list of all possible histories
    @staticmethod
    def build_history_list(file_full_name: str) -> list:
        sentences, tags = Parsing.parse_wtag_file_to_lists(file_full_name)
        histories = []
        for sentence, sentence_tags in zip(sentences, tags):
            for idx, word in enumerate(sentence):
                if idx > 1:
                    previous_tags = (sentence_tags[idx-2], sentence_tags[idx-1])
                    histories.append(History(word, previous_tags))
                else:
                    if idx == 0:
                        histories.append(History(word))
                    else:
                        previous_tags = ("*", sentence_tags[idx-1])
                        histories.append(History(word, previous_tags))
        return histories
