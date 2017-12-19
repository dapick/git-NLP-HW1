from basicModel import BasicModel
from history import TaggedHistory
from consts import Consts

from time import time
import numpy as np

from scipy.sparse import coo_matrix


class Viterbi:
    model = None

    def __init__(self, words: list, model):
        self.model = model
        self.words = words
        self.n = len(self.words)
        self.num_of_tags = len(Consts.POS_TAGS)

        pi = np.zeros([len(Consts.POS_TAGS)] * 2)
        pi[0, 0] = 1
        self.pi = [pi]
        # For each pair of tags holds their bp tag's index. -2 is the idx of the tag before '*', '*'
        self.bp = [np.array([-2], dtype='int32')]

    def q(self, word_idx: int):
        histories = [TaggedHistory([tag1, tag2], self.words, word_idx, tag) for tag1 in Consts.POS_TAGS
                     for tag2 in Consts.POS_TAGS for tag in range(len(Consts.POS_TAGS))]
        data = []
        rows = []
        cols = []
        for history_idx, history in enumerate(histories):
            tag = Consts.POS_TAGS[history.tag_idx]
            list_idxs = self.model.feature.history_matched_features(history, tag)
            len_list_idx = len(list_idxs)
            data += [1] * len_list_idx
            rows += [history_idx] * len_list_idx
            cols += list_idxs
        features_matrix = coo_matrix((data, (rows, cols)),
                                     shape=(len(Consts.POS_TAGS) ** 3, len(self.model.v_parameter))).tocsr()
        inner_sum = np.exp(features_matrix.dot(self.model.v_parameter)).reshape(([len(Consts.POS_TAGS)] * 3))
        denominator = np.sum(inner_sum, axis=-1)
        return inner_sum / denominator

    def run_viterbi(self) -> list:
        # Consts.TIME = 1

        for word_idx in range(1, self.n + 1):
            # t1 = time()
            pi_word_idx = np.zeros((self.num_of_tags, self.num_of_tags))
            bp_word_idx = np.zeros((self.num_of_tags, self.num_of_tags), dtype='int32')
            q = self.q(word_idx-1)
            for u_idx in range(self.num_of_tags):
                for v_idx in range(self.num_of_tags):
                    # TODO: tagging words with '*' somehow.
                    if v_idx != 0:
                        pi_word_idx[u_idx, v_idx] = np.max(np.multiply(self.pi[word_idx - 1][:, u_idx], q[:, u_idx, v_idx]))
                        bp_word_idx[u_idx, v_idx] = np.argmax(np.multiply(self.pi[word_idx - 1][:, u_idx], q[:, u_idx, v_idx]))
            self.pi.append(pi_word_idx)
            self.bp.append(bp_word_idx)
            # Consts.print_time("Tagging word '" + self.words[word_idx - 1] + "', time() - t1)

        idx_row_of_max = np.argmax(self.pi[self.n])
        tag_idx_n_minus_1, tag_idx_n = np.unravel_index(idx_row_of_max, self.pi[self.n].shape)

        res_idx = [tag_idx_n, tag_idx_n_minus_1]

        for word_idx in reversed(range(1, self.n - 1)):
            i = res_idx[-1]
            j = res_idx[-2]
            res_idx.append(self.bp[word_idx + 2][i, j])

        res = [Consts.POS_TAGS[idx] for idx in res_idx]
        return res[::-1]
