import numpy as np
from basicModel import BasicModel
from history import History
import tags


class Viterbi:
    basicModel = None

    # TODO: getting list of words
    def __init__(self, words: list):
        self.basicModel = BasicModel()
        self.words = words
        self.tags = tags.Tags().POS_TAGS

    def q(self, idx: int, tag: str, tag1: str, tag2: str):
        history = History(self.words[idx], (tag1, tag2))
        self.basicModel.probability(history, tag)

    def max_prob(self, pi:  dict, idx:  int, u: str, v: str):
        max_prob = 0
        max_bp = '*'
        for i in range(0, (idx - 1)):
            t = self.tags.POS_TAGS[i]
            cur = (pi[((idx - 1), t, u)] * self.q(idx, v, t, u))
            if max_prob < cur:
                max_prob = cur
                max_bp = [t, u]
        return [max_prob, max_bp]


    def run_viterbi(self):
        # initialization
        num_words = len(self.words)
        pi = {(0, '*', '*'): 1}
        bp = {}
        res = {}

        for k in range(1, num_words-1):
            for i, u in enumerate(self.tags):
                for j in range(0, i):
                    v = self.tags[j]
                    pi[(k, u, v)], bp[(k, u, v)] = self.max_prob(pi, k, u, v)

        for i, u in enumerate(self.tags):
            for j in range(0, i):
                v = self.tags[j]
                max_n, t_n = self.max_prob(pi, num_words, u, v)

        for k in range(0, (num_words-2)):
            res[k] = bp[((k+2), t_n[0], t_n[1])]

        return res


def main():
    pi = {('*', 'D', '*'): 1}
    print(pi)


if '__main__' == __name__:
    main()
