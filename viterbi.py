from time import time

from basicModel import BasicModel
from history import History
from consts import Consts

from math import exp
import numpy as np
from scipy.sparse import coo_matrix


class Viterbi:
    basic_model = None

    def __init__(self, words: list, basic_model: BasicModel):
        self.basic_model = basic_model
        self.words = words
        self.tags = Consts.POS_TAGS
        self.num_of_tags = len(self.tags)

        self.pi = [np.array([1])]
        # For each pair of tags holds their bp tag's index. -2 is the idx of the tag before '*', '*'
        self.bp = [np.array([-2], dtype='int32')]

    def q(self, tag1: str, tag2: str, word_idx: int, tag_idx: int):
        history = History([tag1, tag2], self.words, word_idx)
        return exp(self.basic_model.log_probability(history, tag_idx))

    def _max_prob(self, word_idx: int):

        if word_idx >= 3:
            t1 = time()
            pi_word_idx = np.zeros([self.num_of_tags, self.num_of_tags])
            bp_word_idx = np.zeros([self.num_of_tags, self.num_of_tags], dtype='int32')
            for v_idx, _ in enumerate(Consts.POS_TAGS):
                q_per_v = []
                for t in Consts.POS_TAGS:
                    q_per_v_t = [self.q(t, u, word_idx-1, v_idx) for u in Consts.POS_TAGS]
                    q_per_v.append(q_per_v_t)

                q_per_v = np.array(q_per_v)

                # Calculates pi and bp
                prod_cols = self.pi[word_idx - 1] * q_per_v
                pi_word_idx[:, v_idx] = prod_cols.max(0)
                bp_word_idx[:, v_idx] = prod_cols.argmax(0)
            self.pi.append(pi_word_idx)
            self.bp.append(bp_word_idx)
            Consts.print_time("_max_prob for k = " + str(word_idx), time() - t1)

        elif word_idx == 1:
            t1 = time()
            # Calculates pi and bp
            pi_word_idx = []
            for v_idx, _ in enumerate(Consts.POS_TAGS):
                pi_word_idx.append(self.q('*', '*', word_idx-1, v_idx))
            self.pi.append(np.array([pi_word_idx]))
            self.bp.append(np.array([-1], dtype='int32'))
            Consts.print_time("_max_prob for k = 0", time() - t1)

        elif word_idx == 2:
            t1 = time()
            pi_word_idx = np.zeros([self.num_of_tags, self.num_of_tags])
            bp_word_idx = np.zeros([self.num_of_tags, self.num_of_tags], dtype='int32')
            for v_idx, _ in enumerate(Consts.POS_TAGS):
                q_per_v = []
                q_per_v_t = [self.q('*', u, word_idx - 1, v_idx) for u in Consts.POS_TAGS]
                q_per_v.append(q_per_v_t)

                q_per_v = np.array(q_per_v)

                # Calculates pi and bp
                prod_cols = self.pi[word_idx - 1] * q_per_v
                pi_word_idx[:, v_idx] = prod_cols.max(0)
                bp_word_idx[:, v_idx] = prod_cols.argmax(0)
            self.pi.append(pi_word_idx)
            self.bp.append(bp_word_idx)
            Consts.print_time("_max_prob for k = " + str(word_idx), time() - t1)

    def run_viterbi(self)-> list:
        n = len(self.words)
        Consts.TIME = 1

        for k in range(1, n + 1):
            Consts.print_info("run_viterbi", "Tagging word '" + self.words[k-1] + "'")
            self._max_prob(k)

        idx_row_of_max = (self.pi[n]).argmax()
        tag_idx_n_minus_1, tag_idx_n = np.unravel_index(idx_row_of_max, self.pi[n].shape)

        res_idx = [tag_idx_n, tag_idx_n_minus_1]

        for k in reversed(range(1, n - 1)):
            i = res_idx[-2]
            j = res_idx[-1]
            res_idx.append(self.bp[k + 2][i, j])

        res = [Consts.POS_TAGS[idx] for idx in res_idx]
        return res[::-1]
