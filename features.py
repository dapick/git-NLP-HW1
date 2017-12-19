from history import History, Histories
from consts import Consts

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
    # Dict of {h: list of tags applies}
    tags_per_history = None

    def __init__(self, method: str, model: str, used_features: list=None, file_full_name: str=None):
        if method == Consts.TRAIN:
            self._training(model, used_features, file_full_name)
        elif method == Consts.TAG:
            self._set_internal_values(model)

    def _training(self, model: str, used_features: list, file_full_name: str=Consts.PATH_TO_TRAINING):
        self.features_funcs = {"100": self.feature_100, "101": self.feature_101, "102": self.feature_102,
                               "103": self.feature_103, "104": self.feature_104, "105": self.feature_105}
        self.histories, self.tags = \
            Histories.build_history_list_and_tags_list(file_full_name)
        self.used_features = used_features
        self.idx = 0
        self.feature_vector = {}
        self.features_occurrences = []

        for feature_type in used_features:
            self.features_funcs[feature_type]()

        with open('../data_from_training/advanced_model/feature_vector', 'w+') as f:
            for key, values in self.feature_vector.items():
                print(str(key) + " => " + str(values), file=f)

        # Creates 'history_tag_features'
        self._calculate_history_tag_features()

        if model == Consts.BASIC_MODEL:
            with open('../data_from_training/basic_model/feature_vector', 'w+') as f:
                for key, values in self.feature_vector.items():
                    print(str(key) + " => " + str(values), file=f)
            with open("../data_from_training/basic_model/internal_values_of_feature", 'wb') as f:
                pickle.dump([self.feature_vector, self.used_features], f, protocol=-1)
        elif model == Consts.ADVANCED_MODEL:
            with open("../data_from_training/advanced_model/internal_values_of_feature", 'wb') as f:
                pickle.dump([self.feature_vector, self.used_features], f, protocol=-1)

    def _set_internal_values(self, model: str):
        self.history_tag_features = {}
        if model == Consts.BASIC_MODEL:
            with open("../data_from_training/basic_model/internal_values_of_feature", 'rb') as f:
                self.feature_vector, self.used_features = pickle.load(f)
        elif model == Consts.ADVANCED_MODEL:
            with open("data_from_training/advanced_model/internal_values_of_feature", 'rb') as f:
                self.feature_vector, self.used_features = pickle.load(f)

    # Gives an index for each feature
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

    # Inserts the key idx to 'history_features_idxs' only if the key exist
    def insert_idx(self, key: tuple, history_features_idxs: list):
        feature_value = self.feature_vector.get(key)
        if feature_value:
            history_features_idxs.append(feature_value[0])

    # Calculates a list of features' idx which apply on a certain pair: (h, t)
    def history_matched_features(self, history: History, tag: str):
        history_features_idxs = []
        current_word = history.get_current_word()
        current_word_len = len(current_word)
        # Feature_100
        if "100" in self.used_features:
            self.insert_idx(("100", (current_word, tag)), history_features_idxs)
        # Feature_101 + Feature_102
        if "101" in self.used_features and "102" in self.used_features:
            self.insert_idx(("101", (history.word_custom_suffix(1), tag)), history_features_idxs)
            self.insert_idx(("102", (history.word_custom_prefix(1), tag)), history_features_idxs)
            if current_word_len >= 2:
                self.insert_idx(("101", (history.word_custom_suffix(2), tag)), history_features_idxs)
                self.insert_idx(("102", (history.word_custom_prefix(2), tag)), history_features_idxs)
            if current_word_len >= 3:
                self.insert_idx(("101", (history.word_custom_suffix(3), tag)), history_features_idxs)
                self.insert_idx(("102", (history.word_custom_prefix(3), tag)), history_features_idxs)
            if current_word_len >= 4:
                self.insert_idx(("101", (history.word_custom_suffix(4), tag)), history_features_idxs)
                self.insert_idx(("102", (history.word_custom_prefix(4), tag)), history_features_idxs)

        # Feature 103
        if "103" in self.used_features:
            self.insert_idx(("103", (history.tags[0], history.tags[1], tag)),
                            history_features_idxs)
        # Feature 104
        if "104" in self.used_features:
            self.insert_idx(("104", (history.tags[1], tag)), history_features_idxs)
        # Feature 105
        if "105" in self.used_features:
            self.insert_idx(("105", tag), history_features_idxs)

        return history_features_idxs

    def _calculate_history_tag_features(self):
        Consts.print_info("_calculate_history_tag_features", "Preprocessing")
        self.history_tag_features = {}
        history_tag_list = [(history, tag) for history in self.histories for tag in Consts.POS_TAGS]
        self.tags_per_history = {}
        for history in self.histories:
            self.tags_per_history[history] = []
        for (history, tag) in history_tag_list:
            history_features_idxs = self.history_matched_features(history, tag)
            if history_features_idxs:
                # Saves the (h,t) in the dict 'history_tag_features' only if they apply to some feature
                self.history_tag_features[(history, tag)] = history_features_idxs
                self.tags_per_history[history].append(tag)

    def count_features_types(self):
        for feature_type in self.used_features:
            count_feature = sum([1 if feature_num == feature_type else 0 for feature_num, _ in self.feature_vector])
            Consts.print_info("feature_" + feature_type, "Has " + str(count_feature) + " features")
