class Consts:
    POS_TAGS = ['VBN', 'PRP$', 'PRP', '-LRB-', 'WP', 'PDT', 'VBP', ',', 'NNPS', 'WDT', 'NN', 'EX', '``', 'VBG', 'RP',
                'TO', 'VBZ', 'VBD', 'JJ', 'NNP', 'FW', 'NNS', 'MD', 'CC', 'DT', '$', 'WP$', 'POS', '.', 'VB', 'RB',
                'SYM', ':', 'CD', '-RRB-', 'JJR', 'RBS', 'WRB', '#', 'JJS', 'UH', 'RBR', "''", 'IN']

    PREFIXES = ['a', 'anti', 'back', 'be', 'by', 'co', 'de', 'dis', 'down', 'en', 'em', 'ex', 'fore', 'hind', 'mid',
                'midi', 'mini', 'mis', 'off', 'on', 'out', 'over', 'post', 'pre', 'pro', 're', 'self', 'step', 'twi',
                'un', 'up', 'with', 'a', 'Afro', 'ambi', 'an', 'ana', 'ante', 'anti', 'apo', 'ap', 'arch', 'auto', 'bi',
                'bio', 'cis', 'con', 'co', 'com', 'col', 'cor', 'cryo', 'de', 'demi', 'demo', 'di', 'dia', 'dis', 'di',
                'dif', 'du', 'duo', 'eco', 'en', 'el', 'em', 'epi', 'ep', 'Euro', 'ex', 'geo', 'gyro', 'hemi', 'homo',
                'hypo', 'ideo', 'idio', 'in', 'Indo', 'in', 'il', 'im', 'ir', 'iso', 'macr', 'mal', 'maxi', 'mega',
                'meta', 'mono', 'mon', 'mult', 'neo', 'non', 'omni', 'pan', 'para', 'ped', 'per', 'peri', 'pleo', 'pod',
                'poly', 'post', 'pre', 'pro', 'pro', 'pros', 'pyro', 'semi', 'sub', 'sup', 'sur', 'syn', 'sy', 'syl',
                'sym', 'tele', 'tri', 'uni', 'vice', 'gain', 'umbe', 'y']
    SUFFIXES = ['able', 'ac', 'adeֲ', 'age', 'al', 'an', 'ian', 'ance', 'ant', 'ar', 'ard', 'ary', 'ate', 'cide',
                'crat', 'cule', 'cy', 'dom', 'dox', 'ed', 'ee', 'eer', 'emia', 'en', 'ence', 'ency', 'ent', 'er', 'ern',
                'ese', 'ess', 'est', 'etic', 'ette', 'ful', 'fy', 'gam', 'gamy', 'gon', 'hood', 'ial', 'ian', 'ic',
                'ical', 'ile', 'ily', 'ine', 'ing', 'ion', 'ious', 'ish', 'ism', 'ist', 'ite', 'itis', 'ity', 'ive',
                'ize', 'less', 'let', 'like', 'ling', 'log', 'ly', 'ment', 'ness', 'oid', 'oma', 'onym', 'opia', 'opsy',
                'or', 'ory', 'osis', 'ous', 'path', 'pnea', 'sect', 'ship', 'sion', 'some', 'th', 'tion', 'tome',
                'tomy', 'tude', 'ty', 'ular', 'uous', 'ure', 'ward', 'ware', 'wise', 'y']

    PATH_TO_TRAINING = "data/train.wtag"
    PATH_TO_TEST = "data/test.wtag"
    PATH_TO_COMPETITION = "data/comp.words"

    TRAIN = "Train"
    BASIC_MODEL = "basic_model model"
    ADVANCED_MODEL = "advanced model"

    TAG = "Tag"

    DEBUG = 0
    TIME = 0

    @staticmethod
    def print_info(function_name: str, message: str):
        print("-I-(" + function_name + "): " + message)

    @staticmethod
    def print_debug(function_name: str, message: str):
        if Consts.DEBUG == 1:
            print("-D-(" + function_name + "): " + message)

    @staticmethod
    def print_time(function_name: str, time: float):
        if Consts.TIME == 1:
            print("-T-(" + function_name + "): took " + str(time) + " seconds")
