from basicModel import BasicModel
from history import History
from consts import Consts
from math import exp


class Viterbi:
    basicModel = None

    def __init__(self, words: list):
        self.basicModel = BasicModel(Consts.TAG)
        self.words = words
        self.tags = Consts.POS_TAGS
        self.prev_tags = {-1: [], -2: []}
        self.pi = {-1: {('*', '*'): {"prob": 1, "bp": 'EMPTY'}}, -2: {('EMPTY', '*'): {"prob": 0, "bp": 'EMPTY'}}}

    def q(self, tag1: str, tag2: str, idx: int, tag_idx: int):
        history = History([tag1, tag2], self.words, idx)
        result = exp(self.basicModel.log_probability(history, tag_idx))

        return result

    def _max_prob(self, idx:  int, u: str, tag_idx: int):
        max_prob = 0
        max_bp = ()

        for t in self.prev_tags[-2]:
            # if idx == 0:
            #    return {"prob": 1, "bp": '*'}
            cur = (self.q(t, u, idx, tag_idx) * self.pi[(idx - 1)][(t, u)]["prob"])
            if max_prob < cur:
                max_prob = cur
                max_bp = t
        return {"prob": max_prob, "bp": max_bp}

    def _max_prob_tag_by_idx(self, idx: int, v: str)-> list:
        res = ""
        max_tmp = 0
        for k in self.pi[idx].keys():
            if k[1] != v:
                continue
            if self.pi[idx][k]["prob"] > max_tmp:
                max_tmp = self.pi[idx][k]["prob"]
                res = k[0]

        return res



    def run_viterbi(self)-> list:
        num_words = len(self.words)
        n = num_words - 1
        res = []

        for k in range(0, num_words):
            Consts.print_info("run_viterbi", "Tagging word '" + self.words[k] + "'")
            if k == 0:
                self.prev_tags[-1] = ['*']
                self.prev_tags[-2] = ['*']
            else:
                self.prev_tags[-1].clear()
                self.prev_tags[-2].clear()
                for (u, v) in self.pi[(k - 1)].keys():
                    if v not in self.prev_tags[-1]:
                        self.prev_tags[-1].append(v)
                    if u not in self.prev_tags[-2]:
                        self.prev_tags[-2].append(u)

            for u in self.prev_tags[-1]:
                for j, v in enumerate(self.tags):
                    if k not in self.pi:
                        self.pi[k] = {}
                    self.pi[k][(u, v)] = self._max_prob(k, u, j)

        for k in range(0, n + 1):
            with open("utils/viterbi_pi.txt", "w+") as f:
                print("Key = "+str(k)+"- "+str(self.pi[k]), file=f)

        max_prob_n = 0
        key_max = ('*', '*')
        for key in self.pi[n].keys():
            if self.pi[n][key]["prob"] > max_prob_n:
                max_prob_n = self.pi[n][key]["prob"]
                key_max = key

        res.append(key_max[1])
        res.append(key_max[0])
       # res.append(max_bp_n)

        for k in reversed(range(0, n - 1)):
            i = res[(n - k - 2)]
            j = res[(n - k - 1)]
            res.append(self.pi[(k + 2)][(i, j)]["bp"])


        return res[::-1]
