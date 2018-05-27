import numpy as np
from sem_opt_classic import SEMOptClassic
from sem_model import SEMData, SEMModel
import random
import itertools as it
from functools import reduce
from math import log


def get_regul(name: str = None):
    """

    :param name:
    :return:
    """
    def regul_l1(params):
        """ ridge """
        return sum(np.abs(params)) / len(params)

    def regul_l2(params):
        """ LASSO """
        return np.linalg.norm(params) ** 2

    def regul_zero(params):
        """ none """
        return 0
    if name is None:
        return regul_zero
    elif name.lower() == 'l1' or name.lower() == 'ridge':
        return regul_l1
    elif name.lower() == 'l2' or name.lower() == 'lasso':
        return regul_l2
    else:
        return regul_zero


class SEMCrossVal:
    def __init__(self, mod: SEMModel, data: SEMData, estimator, n_cv):
        """
        Number of cross-valudation
        :param n_cv:
        """

        def partition(lst, n):
            random.shuffle(lst)
            division = len(lst) / float(n)
            return [lst[int(round(division * i)):
                        int(round(division * (i + 1)))] for i in range(n)]

        self.n_cv = n_cv
        self.opts = [SEMOptClassic(mod, data, estimator, 'l2') for _ in range(
            self.n_cv)]

        self.m_profiles = data.m_profiles
        groups = partition(self.m_profiles, self.n_cv)
        self.cv_prof = []
        self.cv_cov = []
        for i, g in enumerate(groups):
            self.cv_prof += [g]
            remain_prof = reduce(lambda x, y: np.concatenate((x, y), axis=0),
                                 [gr for j, gr in enumerate(groups)
                                  if i != j])
            print(remain_prof.shape)
            self.cv_cov += [np.cov(remain_prof, rowvar=False, bias=True)]
            self.opts[i].m_cov = self.cv_cov[i]

    def cv_likelihood(self, params=None):
        """

        :param params:
        :return:
        """

        # def bic(opt, lld, n_profiles):
        #     return log(n_profiles) * (len(opt.params) - len(opt.param_fixed)) \
        #            - lld

        log_likelihood = []
        # bic_all = []
        for opt, profiles in zip(self.opts, self.cv_prof):
            opt.params = np.array(params)
            opt.optimize()
            m_sigma = opt.calculate_sigma(opt.params)
            lld_tmp = opt.ml_norm_log_likelihood(m_sigma, profiles)
            log_likelihood += [opt.ml_norm_log_likelihood(m_sigma, profiles)]
            # bic_all += [bic(opt, lld_tmp, len(profiles))]
        # print(log_likelihood)


        # print(sum(bic_all))

        return reduce(lambda x, y: x + y, log_likelihood)


    def fix_matrix(self, mx_type):
        for opt in self.opts:
            opt.fix_matrix(mx_type)

    def fix_param_zero(self, param_id):
        for opt in self.opts:
            opt.fix_param_zero(param_id)




