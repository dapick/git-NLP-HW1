from history import TaggedHistory, Histories
from consts import Consts

from time import time
import pickle


class Feature(object):
    histories = None
    tags = None
    idx = None
    features_funcs = None
    used_features = None

    # Dict of: {(feature_number, (feature_definition)): [feature_idx, number_of_times_occurs]}
    feature_vector = None
    # List of: in place 'feature_idx' there is the occurrence_number
    features_occurrences = None

    # Dict of {(h, t): list of features applies}
    history_tag_features = None
    all_possible_tagged_histories = None

    def __init__(self, method: str, model: str, used_features: list=None, file_full_name: str=None):
        if method == Consts.TRAIN:
            self._training(model, used_features, file_full_name)
        elif method == Consts.TAG:
            self._set_internal_values(model)

    def _training(self, model: str, used_features: list, file_full_name: str=Consts.PATH_TO_TRAINING):
        self.features_funcs = {"100": self.feature_100, "101": self.feature_101, "102": self.feature_102,
                               "103": self.feature_103, "104": self.feature_104, "105": self.feature_105,
                               "capital_letter": self.feature_capital_letter, "hyphen_word": self.feature_hyphen_word}
        self.used_features = used_features
        self.idx = 0
        self.feature_vector = {}
        self.features_occurrences = []
        self.histories, self.tags = \
            Histories.build_history_list_and_tags_list(file_full_name)
        self.len_histories_in_train = len(self.histories)

        for feature_type in used_features:
            self.features_funcs[feature_type]()

        self._calculate_all_possible_tagged_histories()
        self.len_all_possible_tagged_histories = len(self.all_possible_tagged_histories)

        # Creates 'history_tag_features'
        self._calculate_history_tag_features()

        with open('../data_from_training/' + model + '/feature_vector', 'w+') as f:
            for key, values in self.feature_vector.items():
                print(str(key) + " => " + str(values), file=f)
        with open("../data_from_training/" + model + "/internal_values_of_feature", 'wb') as f:
            pickle.dump([self.feature_vector, self.used_features], f, protocol=-1)

    def _set_internal_values(self, model: str):
        with open("../data_from_training/" + model + "/internal_values_of_feature", 'rb') as f:
            self.feature_vector, self.used_features = pickle.load(f)

    # Gives an index for each feature and count how many time it was used
    def feature_structure(self, keys: tuple):
        if keys not in self.feature_vector:
            self.feature_vector[keys] = [self.idx, 1]
            self.features_occurrences.append(1)
            self.idx += 1
        else:
            self.feature_vector[keys][1] += 1
            self.features_occurrences[self.feature_vector[keys][0]] += 1

    def feature_100(self):
        Consts.print_info("feature_100", "Building")
        for history, tag in zip(self.histories, self.tags):
            self.feature_structure(("100", (history.get_current_word(), tag)))

    def feature_101(self):
        Consts.print_info("feature_101", "Building")
        for history, tag in zip(self.histories, self.tags):
            if history.word_custom_suffix(1) in Consts.SUFFIXES:
                self.feature_structure(("101", (history.word_custom_suffix(1), tag)))
            current_word_len = len(history.get_current_word())
            if current_word_len >= 2 and history.word_custom_suffix(2) in Consts.SUFFIXES:
                self.feature_structure(("101", (history.word_custom_suffix(2), tag)))
            if current_word_len >= 3 and history.word_custom_suffix(3) in Consts.SUFFIXES:
                self.feature_structure(("101", (history.word_custom_suffix(3), tag)))
            if current_word_len >= 4 and history.word_custom_suffix(4) in Consts.SUFFIXES:
                self.feature_structure(("101", (history.word_custom_suffix(4), tag)))

    def feature_102(self):
        Consts.print_info("feature_102", "Building")
        for history, tag in zip(self.histories, self.tags):
            if history.word_custom_prefix(1) in Consts.PREFIXES:
                self.feature_structure(("102", (history.word_custom_prefix(1), tag)))
            current_word_len = len(history.get_current_word())
            if current_word_len >= 2 and history.word_custom_prefix(2) in Consts.PREFIXES:
                self.feature_structure(("102", (history.word_custom_prefix(2), tag)))
            if current_word_len >= 3 and history.word_custom_prefix(3) in Consts.PREFIXES:
                self.feature_structure(("102", (history.word_custom_prefix(3), tag)))
            if current_word_len >= 4 and history.word_custom_prefix(4) in Consts.PREFIXES:
                self.feature_structure(("102", (history.word_custom_prefix(4), tag)))

    def feature_103(self):
        Consts.print_info("feature_103", "Building")
        for history, tag in zip(self.histories, self.tags):
            self.feature_structure(("103", (history.tags[0], history.tags[1], tag)))

    def feature_104(self):
        Consts.print_info("feature_104", "Building")
        for history, tag in zip(self.histories, self.tags):
            self.feature_structure(("104", (history.tags[1], tag)))

    def feature_105(self):
        Consts.print_info("feature_105", "Building")
        for tag in self.tags:
            self.feature_structure(("105", tag))

    def feature_capital_letter(self):
        Consts.print_info("feature_capital_letter", "Building")
        for history, tag in zip(self.histories, self.tags):
            if history.get_current_word()[0].isupper():
                self.feature_structure(("capital_letter", tag))

    def feature_hyphen_word(self):
        Consts.print_info("feature_hyphen_word", "Building")
        for history, tag in zip(self.histories, self.tags):
            if '-' in history.get_current_word():
                self.feature_structure(("hyphen_word", tag))

    def _reduce_features(self):
        Consts.print_info("_reduce_features", "Reducing")
        self.idx = 0
        survived_features = {}
        survived_occurrences = []
        for feature_key, feature_value in self.feature_vector.items():
            if feature_value[1] > 5:
                survived_features[feature_key] = (self.idx, feature_value[1])
                survived_occurrences.append(feature_value[1])
                self.idx += 1
        self.feature_vector = survived_features
        self.features_occurrences = survived_occurrences

    # Inserts the key idx to 'history_features_idxs' only if the there is a feature identified by it
    def insert_idx(self, key: tuple, history_features_idxs: list):
        feature_value = self.feature_vector.get(key)
        if feature_value:
            history_features_idxs.append(feature_value[0])

    # Calculates a list of features' idx which apply on a certain taggedHistory
    def history_matched_features(self, history: TaggedHistory):
        history_features_idxs = []
        current_word = history.get_current_word()
        current_tag = history.get_tag_name()
        current_word_len = len(current_word)
        # Feature_100
        if "100" in self.used_features:
            self.insert_idx(("100", (current_word, current_tag)), history_features_idxs)
        # Feature_101 + Feature_102
        if "101" in self.used_features and "102" in self.used_features:
            self.insert_idx(("101", (history.word_custom_suffix(1), current_tag)), history_features_idxs)
            self.insert_idx(("102", (history.word_custom_prefix(1), current_tag)), history_features_idxs)
            if current_word_len >= 2:
                self.insert_idx(("101", (history.word_custom_suffix(2), current_tag)), history_features_idxs)
                self.insert_idx(("102", (history.word_custom_prefix(2), current_tag)), history_features_idxs)
            if current_word_len >= 3:
                self.insert_idx(("101", (history.word_custom_suffix(3), current_tag)), history_features_idxs)
                self.insert_idx(("102", (history.word_custom_prefix(3), current_tag)), history_features_idxs)
            if current_word_len >= 4:
                self.insert_idx(("101", (history.word_custom_suffix(4), current_tag)), history_features_idxs)
                self.insert_idx(("102", (history.word_custom_prefix(4), current_tag)), history_features_idxs)

        # Feature 103
        if "103" in self.used_features:
            self.insert_idx(("103", (history.tags[0], history.tags[1], current_tag)),
                            history_features_idxs)
        # Feature 104
        if "104" in self.used_features:
            self.insert_idx(("104", (history.tags[1], current_tag)), history_features_idxs)
        # Feature 105
        if "105" in self.used_features:
            self.insert_idx(("105", current_tag), history_features_idxs)

        # Feature capital_letter
        if "capital_letter" in self.used_features:
            self.insert_idx(("capital_letter", current_tag), history_features_idxs)
        # Feature hyphen_word
        if "hyphen_word" in self.used_features:
            self.insert_idx(("hyphen_word", current_tag), history_features_idxs)

        return history_features_idxs

    def _calculate_all_possible_tagged_histories(self):
        Consts.TIME = 1
        t1 = time()
        Consts.print_info("_calculate_all_possible_tagged_histories", "Preprocessing")
        self.all_possible_tagged_histories = []
        for history in self.histories:
            for tag_idx in range(Consts.TAGS_AMOUNT):
                self.all_possible_tagged_histories.append(
                    TaggedHistory(history.tags, history.sentence, history.current_word_idx,
                                  tag_idx))
        Consts.print_time("_calculate_all_possible_tagged_histories", time() - t1)

    def _calculate_history_tag_features(self):
        Consts.TIME = 1
        t1 = time()
        Consts.print_info("_calculate_history_tag_features", "Preprocessing")

        self.history_tag_features = {}
        for tagged_history in self.all_possible_tagged_histories:
            self.history_tag_features[tagged_history] = self.history_matched_features(tagged_history)

        Consts.print_time("_calculate_history_tag_features", time() - t1)

    def count_features_types(self):
        for feature_type in self.used_features:
            count_feature = sum([1 if feature_num == feature_type else 0 for feature_num, _ in self.feature_vector])
            Consts.print_info("feature_" + feature_type, "Has " + str(count_feature) + " features")
