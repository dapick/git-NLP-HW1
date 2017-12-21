from consts import Consts
from viterbi import Viterbi
from parsing import Parsing

from model import BasicModel, AdvancedModel
from multiprocessing.pool import Pool
from time import time


class Tagger:
    # input file
    word_file = None
    sentences_list = None

    # output file
    tagged_file = None
    tags_list = None

    # tagger
    viterbi = None

    model = None

    def __init__(self, word_file: str, model: str):
        self.model_type = model

        self.word_file = word_file
        self.sentences_list = Parsing.parse_words_file_to_list(self.word_file)
        if model == Consts.BASIC_MODEL:
            self.model = BasicModel(Consts.TAG)
        elif model == Consts.ADVANCED_MODEL:
            self.model = AdvancedModel(Consts.TAG)

        self.tagged_file = self._get_tagged_file_name()
        self.tags_list = []

    def _get_tagged_file_name(self):
        ret_file = self.word_file
        file_name = ret_file.split('.words')
        if self.model_type == Consts.BASIC_MODEL:
            ret_file = file_name[0] + "_m1_"
        elif self.model_type == Consts.ADVANCED_MODEL:
            ret_file = file_name[0] + "_m2_"
        ret_file = ret_file + "302988217.wtag"
        return ret_file

    def _tag_sentence(self, sentence_tuple: tuple):
        Consts.TIME = 1
        t1 = time()

        sentence_idx, sentence = sentence_tuple
        self.viterbi = Viterbi(sentence, self.model)
        tags = self.viterbi.run_viterbi()

        Consts.print_time("Tagging sentence " + str(sentence_idx + 1), time() - t1)
        return tags

    def tag(self):
        Consts.print_info("tag_file", "Tagging file " + self.word_file)
        t1 = time()

        # Run parallel - good when checking many sentences
        with Pool(6) as pool:
            sentences_tags = pool.map(self._tag_sentence, enumerate(self.sentences_list))

        # Run linear - good when checking a few sentences
        # for sen in self.sentences_list:
        #     self._tag_sentence(sen)

        Parsing.parse_lists_to_wtag_file(self.sentences_list, list(sentences_tags), self.tagged_file)
        Consts.print_time("Tagging file", time() - t1)

    def calculate_accuracy(self, out_file: str, expected_file: str):
        out_list_w, out_ = Parsing.parse_wtag_file_to_lists(out_file)
        exp_list_w, exp_list_t = Parsing.parse_wtag_file_to_lists(expected_file)

        confused_tags = {}
        confused_tags_with_sen = {}
        for tag in Consts.POS_TAGS:
            confused_tags[tag] = {"dict": {}, "sum_wrong": 0}
            confused_tags_with_sen[tag] = []

        count = 0
        for sen_idx, sen in enumerate(out_list_w):
            for word_idx, w in enumerate(sen):
                exp_tag = exp_list_t[sen_idx][word_idx]
                out_tag = out_[sen_idx][word_idx]
                if out_tag not in confused_tags[exp_tag]["dict"]:
                    confused_tags[exp_tag]["dict"][out_tag] = 1
                else:
                    confused_tags[exp_tag]["dict"][out_tag] += 1
                if w == exp_list_w[sen_idx][word_idx] and out_tag == exp_tag:
                    count += 1
                else:
                    confused_tags_with_sen[exp_tag].append({out_tag: (sen, (word_idx + 1), w)})
                    confused_tags[exp_tag]["sum_wrong"] += 1

        with open("../data_from_training/" + self.model_type + "/confusion_matrix", "w+") as f:
            for key, value in sorted(confused_tags.items(), key=lambda x: x[0]):
                print(key + " => " + str(value["dict"]) + "\n" + key + " => " + str(value["sum_wrong"]), file=f)

        list_max_wrong_tags = sorted(sorted(confused_tags.items(), key=lambda x: x[1]["sum_wrong"], reverse=True)[:10],
                                     key=lambda x: x[0])
        with open("../data_from_training/" + self.model_type + "/10_worst_tags", "w+") as f:
            for x in list_max_wrong_tags:
                print(x[0] + " => " + str(sorted(x[1]["dict"].items(), key=lambda x: x[0])) + "\n" + x[0] + " => " + str(x[1]["sum_wrong"]), file=f)

        with open("../data_from_training/" + self.model_type + "/10_worst_tags_with_wrong_sentences", "w+") as f:
            for x in list_max_wrong_tags:
                for tup in confused_tags_with_sen[x[0]]:
                    print(x[0] + " => " + str(tup), file=f)

        num_words = sum(len(out_list_w[k]) for k in range(len(out_list_w)))
        str_res = str(100 * (count / num_words)) + "%"
        print("The accuracy is: " + str_res)
