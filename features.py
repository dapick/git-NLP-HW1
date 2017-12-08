from history import History


class Feature(object):
    @staticmethod
    def feature_100(history: History, tag: str) -> bool:
        # TODO: consider using enums to avoid typos
        return history.current_word == "base" and tag == "Vt"

    @staticmethod
    def feature_101(history: History, tag: str) -> bool:
        # TODO: consider using enums to avoid typos
        return history.current_word.endswith("ing") and tag == "VBG"

    @staticmethod
    def feature_102(history: History, tag: str) -> bool:
        # TODO: consider using enums to avoid typos
        return history.current_word.startswith(("pre", "Pre")) and tag == "NN"

    @staticmethod
    def feature_103(history: History, tag: str) -> bool:
        # TODO: consider using enums to avoid typos
        return (history.tags, tag) == (("DT", "JJ"), "Vt")

    @staticmethod
    def feature_104(history: History, tag: str) -> bool:
        # TODO: consider using enums to avoid typos
        return (history.tags[1], tag) == ("JJ", "Vt")

    @staticmethod
    def feature_105(history: History, tag: str) -> bool:
        # TODO: consider using enums to avoid typos
        # TODO: consider not sending history as a parameter
        return tag == "Vt"
