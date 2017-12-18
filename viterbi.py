from time import time

from basicModel import BasicModel
from history import History
from consts import Consts

from math import exp
import numpy as np

from tagger import Tagger
# from scipy import sparse
# from scipy.sparse import csr_matrix, lil_matrix

# TODO: Delete, only for testing
import pickle


class Viterbi:
    basic_model = None

    def __init__(self, words: list, basic_model: BasicModel):
        self.basic_model = basic_model
        self.words = words
        self.n = len(self.words)
        self.num_of_tags = len(Consts.POS_TAGS)

        self.tags_per_word = Tagger.tags_per_word

        self.pi = [np.asarray([1])]
        # For each pair of tags holds their bp tag's index. -2 is the idx of the tag before '*', '*'
        self.bp = [np.asarray([-2], dtype='int32')]
        # List of lists of coo_matrix
        # For each word_idx holds a list of coo_matrix, one for each possible tag
        # e.g.: q_values[word_idx][tag_idx] = coo_matrix for every t,u tags with tag 'tag_idx'
        self.q_values = []
        self._calculate_all_q()
        # TODO: delete. just for testing
        # with open("../utils/q_values", 'rb') as f:
        #     self.q_values = pickle.load(f)

    def q(self, tag1: str, tag2: str, word_idx: int, tag_idx: int):
        history = History([tag1, tag2], self.words, word_idx)
        return exp(self.basic_model.log_probability(history, tag_idx))

    def _calculate_all_q(self):
        Consts.print_info("_calculate_all_q", "Preprocessing")
        Consts.TIME = 1
        t1 = time()
        for k in range(0, self.n):
            list_of_tags_matrices = []
            for v_idx in range(0, self.num_of_tags):
                if k >= 2:
                    matrix_for_v = np.zeros((self.num_of_tags, self.num_of_tags))
                    for t_idx, t in enumerate(Consts.POS_TAGS):
                        for u_idx, u in enumerate(Consts.POS_TAGS):
                            matrix_for_v[t_idx, u_idx] = self.q(t, u, k, v_idx)
                    list_of_tags_matrices.append(matrix_for_v)

                elif k == 0:
                    list_of_tags_matrices.append(np.asarray(self.q('*', '*', k, v_idx)))

                elif k == 1:
                    matrix_for_v = np.zeros((1, self.num_of_tags))
                    for u_idx, u in enumerate(Consts.POS_TAGS):
                        matrix_for_v[0, u_idx] = self.q('*', u, k, v_idx)
                    list_of_tags_matrices.append(matrix_for_v)

            self.q_values.append(list_of_tags_matrices)

        Consts.print_time("_calculate_all_q", time() - t1)
        # TODO: delete. just for testing
        with open("../utils/q_values", 'wb') as f:
            pickle.dump(self.q_values, file=f, protocol=-1)

    def _max_prob(self, word_idx: int):
        if word_idx >= 3:
            t1 = time()
            pi_word_idx = np.zeros((self.num_of_tags, self.num_of_tags))
            bp_word_idx = np.zeros((self.num_of_tags, self.num_of_tags), dtype='int32')
            for v_idx, _ in enumerate(Consts.POS_TAGS):
                # Calculates pi and bp
                prod_cols = self.pi[word_idx - 1] * self.q_values[word_idx-1][v_idx]
                pi_word_idx[:, v_idx] = prod_cols.max(0)
                bp_word_idx[:, v_idx] = prod_cols.argmax(0)
            self.pi.append(pi_word_idx)
            self.bp.append(bp_word_idx)
            Consts.print_time("_max_prob for k = " + str(word_idx), time() - t1)

        elif word_idx == 1:
            self.pi.append(self.q_values[word_idx-1])
            self.bp.append(np.array([-1], dtype='int32'))

        elif word_idx == 2:
            t1 = time()
            pi_word_idx = np.zeros((self.num_of_tags, self.num_of_tags))
            bp_word_idx = np.zeros((self.num_of_tags, self.num_of_tags), dtype='int32')
            for v_idx, _ in enumerate(Consts.POS_TAGS):
                # Calculates pi and bp
                prod_cols = self.pi[word_idx - 1] * self.q_values[word_idx - 1][v_idx]
                pi_word_idx[:, v_idx] = prod_cols.max(0)
                bp_word_idx[:, v_idx] = prod_cols.argmax(0)
            self.pi.append(pi_word_idx)
            self.bp.append(bp_word_idx)
            Consts.print_time("_max_prob for k = " + str(word_idx), time() - t1)

    def run_viterbi(self) -> list:
        Consts.TIME = 1

        for k in range(1, self.n + 1):
            Consts.print_info("run_viterbi", "Tagging word '" + self.words[k-1] + "'")
            self._max_prob(k)

        idx_row_of_max = (self.pi[self.n]).argmax()
        tag_idx_n_minus_1, tag_idx_n = np.unravel_index(idx_row_of_max, self.pi[self.n].shape)

        res_idx = [tag_idx_n, tag_idx_n_minus_1]

        for k in reversed(range(1, self.n - 1)):
            i = res_idx[-2]
            j = res_idx[-1]
            res_idx.append(self.bp[k + 2][i, j])

        res = [Consts.POS_TAGS[idx] for idx in res_idx]
        return res[::-1]


    