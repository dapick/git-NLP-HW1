import numpy as np
from consts import Consts


class Aux:

    @property
    def unique_tags_from_train_file(self):
        tags = set()
        tag_file = Consts.PATH_TO_TRAINING
        with open(tag_file, 'r') as f:
            for line in f:
                for word in line.split():
                    _, t = word.split("_")
                    tags.add(t)
        return tags


print(Aux().unique_tags_from_train_file)





