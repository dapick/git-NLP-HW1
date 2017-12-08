import numpy as np


class Aux:

    @property
    def unique_tags_from_train_file(self):
        tags = []
        tag_file = "data/train.wtag"
        with open(tag_file, 'r') as f:
            for line in f:
                for word in line.split():
                    [w, t] = word.split("_")
                    tags.append(t)
        tags = np.unique(tags)
        return tags





