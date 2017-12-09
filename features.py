from history import History, Histories
from consts import Consts


class Feature(object):
    histories = None
    tags = None
    idx = None
    features_funcs = None
    used_features = None

    # key = (feature_number, (feature_definition))
    # value = [feature_idx, number_of_times_occurs]
    feature_vector = None
    # The number of occurrences for each feature
    # Dict of {(feature_idx): occurrence_number}
    features_occurrences = None

    # List of features' idx which apply on a certain pair: (h, t)
    history_features_idxs = None

    def __init__(self, file_full_name: str=Consts.PATH_TO_TRAINING,
                 used_features: tuple=("100", "101", "102", "103", "104", "105")):
        self.histories, self.tags = \
            Histories.build_history_list_and_tags_list(file_full_name)
        self.used_features = used_features

        self.features_funcs = {"100": self.feature_100, "101": self.feature_101, "102": self.feature_102,
                               "103": self.feature_103, "104": self.feature_104, "105": self.feature_105}

        self.idx = 0
        self.feature_vector = {}
        self.features_occurrences = {}

        for feature_type in used_features:
            self.features_funcs[feature_type]()

        self._reduce_features()

        self.history_features_idxs = []

    # Gives an index for each feature
    def feature_structure(self, keys: tuple):
        if keys not in self.feature_vector:
            self.feature_vector[keys] = [self.idx, 1]
            self.features_occurrences[self.idx] = 1
            self.idx += 1
        else:
            self.feature_vector[keys][1] += 1
            self.features_occurrences[self.feature_vector[keys][0]] += 1

    def feature_100(self):
        Consts.print_status("feature_100", "Building")
        for history, tag in zip(self.histories, self.tags):
            self.feature_structure(("100", (history.get_current_word().lower(), tag)))

    def feature_101(self):
        Consts.print_status("feature_101", "Building")
        for history, tag in zip(self.histories, self.tags):
            self.feature_structure(("101", (history.word_custom_suffix(1).lower(), tag)))
            current_word_len = len(history.get_current_word())
            if current_word_len >= 2:
                self.feature_structure(("101", (history.word_custom_suffix(2).lower(), tag)))
            if current_word_len >= 3:
                self.feature_structure(("101", (history.word_custom_suffix(3).lower(), tag)))
            if current_word_len >= 4:
                self.feature_structure(("101", (history.word_custom_suffix(4).lower(), tag)))

    def feature_102(self):
        Consts.print_status("feature_102", "Building")
        for history, tag in zip(self.histories, self.tags):
            self.feature_structure(("102", (history.word_custom_prefix(1).lower(), tag)))
            current_word_len = len(history.get_current_word())
            if current_word_len >= 2:
                self.feature_structure(("102", (history.word_custom_prefix(2).lower(), tag)))
            if current_word_len >= 3:
                self.feature_structure(("102", (history.word_custom_prefix(3).lower(), tag)))
            if current_word_len >= 4:
                self.feature_structure(("102", (history.word_custom_prefix(4).lower(), tag)))

    def feature_103(self):
        Consts.print_status("feature_103", "Building")
        for history, tag in zip(self.histories, self.tags):
            self.feature_structure(("103", (history.tags[0], history.tags[1], tag)))

    def feature_104(self):
        Consts.print_status("feature_104", "Building")
        for history, tag in zip(self.histories, self.tags):
            self.feature_structure(("104", (history.tags[1], tag)))

    def feature_105(self):
        Consts.print_status("feature_105", "Building")
        for tag in self.tags:
            self.feature_structure(("105", tag))

    def _reduce_features(self):
        Consts.print_status("_reduce_features", "Reducing")
        self.idx = 0
        survived_features = {}
        survived_occurrences = {}
        for feature_key, feature_value in self.feature_vector.items():
            if feature_value[1] > 5:
                survived_features[feature_key] = (self.idx, feature_value[1])
                survived_occurrences[self.idx] = feature_value[1]
                self.idx += 1
        self.feature_vector = survived_features
        self.features_occurrences = survived_occurrences

    # Inserts the key idx to 'history_features_idxs' only if the key exist
    def insert_idx(self, key: tuple):
        feature_value = self.feature_vector.get(key)
        if feature_value:
            self.history_features_idxs.append(feature_value[0])

    # Returns the list 'history_features_idxs' updated to the given (h, t) pair
    def history_matched_features(self, history: History, tag: str) -> list:
        self.history_features_idxs = []
        current_word_len = len(history.get_current_word())

        # Feature_100
        if "100" in self.used_features:
            self.insert_idx(("100", (history.get_current_word().lower(), tag)))
        # Feature_101 + Feature_102
        if "101" in self.used_features and "102" in self.used_features:
            self.insert_idx(("101", (history.word_custom_suffix(1).lower(), tag)))
            self.insert_idx(("102", (history.word_custom_prefix(1).lower(), tag)))
            if current_word_len >= 2:
                self.insert_idx(("101", (history.word_custom_suffix(2).lower(), tag)))
                self.insert_idx(("102", (history.word_custom_prefix(2).lower(), tag)))
            if current_word_len >= 3:
                self.insert_idx(("101", (history.word_custom_suffix(3).lower(), tag)))
                self.insert_idx(("102", (history.word_custom_prefix(3).lower(), tag)))
            if current_word_len >= 4:
                self.insert_idx(("101", (history.word_custom_suffix(4).lower(), tag)))
                self.insert_idx(("102", (history.word_custom_prefix(4).lower(), tag)))
        # Feature 103
        if "103" in self.used_features:
            self.insert_idx(("103", (history.tags[0], history.tags[1], tag)))
        # Feature 104
        if "104" in self.used_features:
            self.insert_idx(("104", (history.tags[1], tag)))
        # Feature 105
        if "105" in self.used_features:
            self.insert_idx(("105", tag))

        return self.history_features_idxs
