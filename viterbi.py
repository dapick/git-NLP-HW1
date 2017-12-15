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

    def q(self, tag1: str, tag2: str, idx: int, tag_idx: int):
        history = History([tag1, tag2], self.words, idx)
        result = exp(self.basicModel.log_probability(history, tag_idx))
        # print("(q) :" + str(result))
        return result

    def _max_prob(self, pi:  dict, idx:  int, u: str, tag_idx: int):
        max_prob = 0
        max_bp = ()
        for t in prev_tags[-2]:
            cur = (self.q(t, u, idx, tag_idx) * pi[(idx - 1)][(t, u)]["prob"])
            if max_prob < cur:
                max_prob = cur
                max_bp = t
        return {"prob": max_prob, "bp": max_bp}

    def _end_of_path(self, pi: dict, n: int)-> list:
        # max (pi[ (n, u, v) ] for u, v in zip (self.tags, self.tags))

        return [max(pi[(n, u, v)], key=pi.get) for u, v in zip(self.tags, self.tags)]


    def run_viterbi(self)-> list:
        # initialization
        num_words = len(self.words)
        n = num_words - 1
        pi = {0: {('*', '*'): {"prob": 1, "bp": 'EMPTY'}}}
        prev_tags = {-1: ['*'], -2: ['*']}
        res = []

        # TODO: inside for loop here
        # for (u, v) in pi[k-1].keys():
        #    prev_tags[-1].append(v)
        #    prev_tags[-2].append(u)

        for k in range(0, n):
            Consts.print_info("run_viterbi", "Tagging word '" + self.words[k] + "'")
            # for i, u in enumerate(prev_tags[-1]):
            for u in self.tags:
                for j, v in enumerate(self.tags):
                    if k not in pi:
                        pi[k] = {}
                    pi[k][(u, v)] = self._max_prob(pi, k, v, j)

        Consts.print_info("run_viterbi", "Tagging word '" + self.words[n] + "'")
        for i, u in enumerate(self.tags):
            for j in range(0, i):
                v = self.tags[j]
                if n not in pi:
                    pi[n] = {}
                pi[n][(u, v)] = self._max_prob(pi, n, u, j)

        max_prob_n = 0
        key_max = ('*', '*')
        for key in pi[n].keys():
            if pi[n][key]["prob"] > max_prob_n:
                max_prob_n = pi[n][key]["prob"]
                max_bp_n = pi[n][key]["bp"]
                key_max = key

        print("max_prob_n = " + str(max_prob_n) + " max_bp_n = " + str(max_bp_n) + " max_key = " + str(key_max))

        print(pi[n])
        # TODO: func to find max bp u,v index n
        for k in reversed(range(0, (num_words-2))):
            res.append(bp[((k+2), t_n[0], t_n[1])])

        print(res)

        return reversed(res)
