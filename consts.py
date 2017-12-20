class Consts:
    POS_TAGS = ['*', '#', '$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS',
                'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO',
                'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``']

    SUFFIXES = ['a', 'ac', 'ad', 'ade', 'age', 'al', 'all', 'an', 'ana', 'ane', 'ant', 'ar', 'ard', 'art', 'ary', 'ase',
                'ate', 'b', 'bot', 'c', 'cy', 'd', 'dom', 'e', 'ean', 'ed', 'ee', 'eer', 'ein', 'eme', 'en', 'ene',
                'ent', 'er', 'es', 'ese', 'ess', 'est', 'et', 'eth', 'ey', 'fer', 'fic', 'ful', 'fy', 'g', 'gen', 'gon',
                'h', 'i', 'ia', 'ial', 'ian', 'ic', 'ick', 'ics', 'id', 'ide', 'ie', 'ied', 'ies', 'ify', 'ile', 'in',
                'ine', 'ing', 'ion', 'ise', 'ish', 'ism', 'ist', 'ite', 'ity', 'ium', 'ive', 'ix', 'ize', 'kin', 'l',
                'le', 'let', 'log', 'ly', 'm', 'mer', 'mo', 'n', "n't", 'o', 'ode', 'oid', 'ol', 'ole', 'oma', 'ome',
                'on', 'one', 'ont', 'or', 'ory', 'ose', 'ous', 'p', 'ped', 'pod', 'q', 'r', 'ric', 'ry', 's', 'st', 't',
                'th', 'ty', 'u', 'ught', 'ule', 'ure', 'x', 'y', 'yl', 'yne']

    PREFIXES = ['a', 'ab', 'abs', 'ac', 'acr', 'acro', 'acti', 'ad', 'aer', 'aero', 'af', 'afte', 'ag', 'agr', 'agri',
                'agro', 'al', 'allo', 'ambi', 'an', 'ana', 'and', 'andr', 'angl', 'ano', 'ant', 'anth', 'anti', 'ap',
                'apo', 'aqui', 'arc', 'arch', 'aris', 'arit', 'arth', 'at', 'atto', 'audi', 'aut', 'auto', 'bact',
                'bar', 'baro', 'bath', 'be', 'bi', 'bibl', 'bin', 'bio', 'blas', 'brac', 'brad', 'by', 'carb', 'card',
                'carp', 'cel', 'cen', 'cent', 'chal', 'chem', 'chin', 'chir', 'chlo', 'chol', 'chri', 'chro', 'chry',
                'circ', 'co', 'col', 'com', 'con', 'cont', 'cor', 'cosm', 'coun', 'cycl', 'cyn', 'de', 'dec', 'deca',
                'deci', 'demi', 'deut', 'di', 'dia', 'dini', 'dipl', 'dis', 'down', 'dys', 'eco', 'elec', 'em', 'en',
                'end', 'endo', 'ent', 'epi', 'equi', 'ethn', 'eu', 'eur', 'euro', 'ever', 'ex', 'exa', 'exo', 'extr',
                'ferr', 'fluo', 'for', 'fore', 'fort', 'fran', 'ful', 'full', 'gain', 'gen', 'geo', 'gymn', 'half',
                'heli', 'hem', 'hemi', 'hemo', 'hend', 'hist', 'hol', 'home', 'hydr', 'hype', 'hypo', 'il', 'ill', 'im',
                'in', 'ind', 'indo', 'inte', 'intr', 'ir', 'is', 'iso', 'ital', 'ker', 'kilo', 'like', 'lip', 'lipo',
                'macr', 'mal', 'mani', 'many', 'mega', 'mes', 'meta', 'metr', 'micr', 'mid', 'mill', 'mini', 'mis',
                'mon', 'mono', 'mult', 'myri', 'new', 'non', 'oct', 'octo', 'off', 'omni', 'on', 'othe', 'out', 'ov',
                'over', 'pale', 'par', 'para', 'pent', 'phon', 'phot', 'phys', 'poly', 'post', 'pre', 'pret', 'prot',
                'psyc', 'quad', 'quar', 'quin', 'radi', 're', 'robo', 'same', 'self', 'semi', 'sept', 'sex', 'span',
                'step', 'sub', 'sui', 'supe', 'supr', 'syl', 'sym', 'syn', 'tele', 'ter', 'ther', 'thor', 'to', 'tran',
                'tri', 'twi', 'ultr', 'un', 'unde', 'uni', 'up', 'ur', 'vice', 'wan', 'well', 'with', 'y']

    PATH_TO_TRAINING = "data/train.wtag"
    PATH_TO_TEST_WTAG = "data/test.wtag"
    PATH_TO_TEST_WORDS = "data/test.words"
    PATH_TO_COMPETITION = "data/comp.words"

    TRAIN = "Train"
    TAG = "Tag"

    BASIC_MODEL = "basic_model"
    ADVANCED_MODEL = "advanced_model"

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
