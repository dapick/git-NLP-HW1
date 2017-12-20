from consts import Consts
from parsing import Parsing


class Aux:

    @property
    def unique_tags_from_train_file(self):
        tags = set()
        tag_file = "../" + Consts.PATH_TO_TRAINING
        with open(tag_file, 'r') as f:
            for line in f:
                for word in line.split():
                    _, t = word.split("_")
                    tags.add(t)
        return sorted(tags)

    @property
    def unique_suffixes_and_prefixes(self):
        sentences, _ = Parsing().parse_wtag_file_to_lists("../" + Consts.PATH_TO_TRAINING)
        words_in_train = set()
        for sentence in sentences:
            for word in sentence:
                words_in_train.add(word.lower())

        with open("prefix.txt") as p:
            prefixes = {line.rstrip().lower()[:4] for line in p.readlines()}

        with open("suffix.txt") as s:
            suffixes = set()
            for word in s.readlines():
                if len(word) > 4:
                    suffixes.add(word[-4].rstrip().lower())
                else:
                    suffixes.add(word.rstrip().lower())

        prefixes_in_train = set()
        suffixes_in_train = set()
        for word in words_in_train:
            for prefix in prefixes:
                if word.startswith(prefix):
                    prefixes_in_train.add(prefix)
            for suffix in suffixes:
                if word.endswith(suffix):
                    suffixes_in_train.add(suffix)

        return sorted(prefixes_in_train), sorted(suffixes_in_train)


if __name__ == "__main__":
    tags = Aux().unique_tags_from_train_file
    print("tags:", tags, "\namount:", len(tags))
    # prefixes_output, suffixes_output = Aux().unique_suffixes_and_prefixes
    # print("prefixes:", prefixes_output, "\namount:", len(prefixes_output))
    # print("suffixes:", suffixes_output, "\namount:", len(suffixes_output))
