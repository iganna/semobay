import numpy as np


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

