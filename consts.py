class Consts:
    POS_TAGS = {'VBN', 'PRP$', 'PRP', '-LRB-', 'WP', 'PDT', 'VBP', ',', 'NNPS', 'WDT', 'NN', 'EX', '``', 'VBG', 'RP',
                'TO', 'VBZ', 'VBD', 'JJ', 'NNP', 'FW', 'NNS', 'MD', 'CC', 'DT', '$', 'WP$', 'POS', '.', 'VB', 'RB',
                'SYM', ':', 'CD', '-RRB-', 'JJR', 'RBS', 'WRB', '#', 'JJS', 'UH', 'RBR', "''", 'IN'}

    PATH_TO_TRAINING = "data/train.wtag"
    PATH_TO_TEST = "data/test.wtag"
    PATH_TO_COMPETITION = "data/comp.words"

    @staticmethod
    def print_status(function_name: str, message: str):
        print("-I-(" + function_name + "): " + message)
