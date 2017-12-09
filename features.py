from history import History, Histories
from consts import Consts

from multiprocessing.pool import Pool
from functools import partial


class Feature(object):
    histories = None
    tags = None
    idx = None
    features_funcs = None
    used_features = None

    # Dict of: {(feature_number, (feature_definition)): [feature_idx, number_of_times_occurs]}
    feature_vector = None
    # Dict of {(feature_idx): occurrence_number}
    features_occurrences = None
    # Dict of {(h, t): list of features applies}
    history_tag_features = None

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

        self.history_tag_features = {}
        # Updates 'history_tag_features'
        self._calculate_history_tag_features()

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
        Consts.print_info("feature_100", "Building")
        for history, tag in zip(self.histories, self.tags):
            self.feature_structure(("100", (history.get_current_word().lower(), tag)))

    def feature_101(self):
        Consts.print_info("feature_101", "Building")
        for history, tag in zip(self.histories, self.tags):
            current_word = history.get_current_word().lower()
            for suffix in Consts.SUFFIXES:
                if current_word.endswith(suffix):
                    self.feature_structure(("101", (suffix, tag)))

    def feature_102(self):
        Consts.print_info("feature_102", "Building")
        for history, tag in zip(self.histories, self.tags):
            current_word = history.get_current_word().lower()
            for prefix in Consts.PREFIXES:
                if current_word.startswith(prefix):
                    self.feature_structure(("102", (prefix, tag)))

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
        survived_occurrences = {}
        for feature_key, feature_value in self.feature_vector.items():
            if feature_value[1] > 5:
                survived_features[feature_key] = (self.idx, feature_value[1])
                survived_occurrences[self.idx] = feature_value[1]
                self.idx += 1
        self.feature_vector = survived_features
        self.features_occurrences = survived_occurrences

    # Inserts the key idx to 'history_features_idxs' only if the key exist
    def insert_idx(self, key: tuple, history_features_idxs: list):
        feature_value = self.feature_vector.get(key)
        if feature_value:
            history_features_idxs.append(feature_value[0])

    # Calculates a list of features' idx which apply on a certain pair: (h, t)
    def history_matched_features(self, history_tag: tuple) -> (tuple, list):
        history_features_idxs = []
        current_word = history_tag[0].get_current_word().lower()
        # Feature_100
        if "100" in self.used_features:
            self.insert_idx(("100", (current_word, history_tag[1])), history_features_idxs)
        # Feature_101
        if "101" in self.used_features:
            for suffix in Consts.SUFFIXES:
                if current_word.endswith(suffix):
                    self.insert_idx(("101", (suffix, history_tag[1])), history_features_idxs)
        # Feature_102
        if "102" in self.used_features:
            for prefix in Consts.PREFIXES:
                if current_word.startswith(prefix):
                    self.insert_idx(("102", (prefix, history_tag[1])), history_features_idxs)
        # Feature 103
        if "103" in self.used_features:
            self.insert_idx(("103", (history_tag[0].tags[0], history_tag[0].tags[1], history_tag[1])),
                            history_features_idxs)
        # Feature 104
        if "104" in self.used_features:
            self.insert_idx(("104", (history_tag[0].tags[1], history_tag[1])), history_features_idxs)
        # Feature 105
        if "105" in self.used_features:
            self.insert_idx(("105", history_tag[1]), history_features_idxs)

        # return history_tag[0], history_tag[1], history_features_idxs
        self.history_tag_features[(history_tag[0], history_tag[1])] = history_features_idxs

    # def _insert_history_tag_features_idxs(self, result: ApplyResult):
    #     history, tag, features_idxs = result
    #     self.history_tag_features[(history, tag)] = features_idxs

    def _calculate_history_tag_features(self):
        Consts.print_info("_calculate_history_tag_features", "Preprocessing")
        # jobs_pool = Pool(4)
        history_tag_list = [(history, tag) for history in self.histories for tag in Consts.POS_TAGS]
        # self.history_tag_features = jobs_pool.apply_async(self.history_matched_features, history_tag_list,
        #                                                   callback=self._insert_history_tag_features_idxs)
        # jobs_pool.close()
        # jobs_pool.join()
        # map(self.history_matched_features, history_tag_list)
        for history_tag_tuple in history_tag_list:
            self.history_matched_features(history_tag_tuple)
        # with open("trialDataFiles/outputAllFeatures.output", 'w') as f:
        #     # print(self.history_tag_features, file=f)
        #     print("Hello World", file=f)
