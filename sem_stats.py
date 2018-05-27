from scipy.special import gammaln
from collections import namedtuple
from sem_opt_abc import SEMOptABC
from sem_model import SEMData, SEMModel
from scipy.stats import norm
from itertools import product
import numpy as np

ParameterStatistics = namedtuple('ParametersStatistics',
                                 ['value', 'se', 'zscore', 'pvalue'])
SEMStatistics = namedtuple('SEMStatistics', ['dof', 'ml', 'aic', 'bic',
                                             'params'])

def calculate_dof(opt: SEMOptABC, data: SEMData):
    p = data.m_cov.shape[0] #Num of observed variables.
    nCov = p * (p + 1) / 2
    return nCov - len(opt.params)

def calculate_likelihood(opt: SEMOptABC, data: SEMData, params=None):
    if params is None:
        params = opt.params
    Sigma = opt.calculate_sigma(params)
    Cov = opt.m_cov
    det_ratio = np.linalg.det(Sigma) / np.linalg.det(Cov)
    log_det_ratio = np.log(det_ratio + 1e-16) if det_ratio > 0 else 1e6
    inv_Sigma = np.linalg.pinv(Sigma)
    loss = np.trace(Cov @ inv_Sigma) + log_det_ratio - Cov.shape[0]
    return loss
#    n, p = calculate_dof(opt, data), len(opt.params)
#    Sigma, S = opt.calculate_sigma(), data.m_cov
#    tr = -n * np.trace(S @ np.linalg.pinv(Sigma)) / 2
#    ld = -n * np.log(np.linalg.det(Sigma)) / 2
#    return tr + ld


def calculate_aic(opt: SEMOptABC, data: SEMData, lh=None):
    if lh is None:
        lh = calculate_likelihood(opt, data)
    return 2 * (len(opt.params) - np.log(lh))


def calculate_bic(opt: SEMOptABC, data: SEMData, lh=None):
    if lh is None:
        lh = calculate_likelihood(opt, data)
    k, n  = len(opt.params), data.m_profiles.shape[0]
    return np.log(n) * k - 2 * np.log(lh)


def calculate_standard_errors(opt: SEMOptABC, data: SEMData, information='expected'):
    def calculate_hessian(params):
        sGrad = opt.calculate_sigma_gradient(params)
        sHess = opt.calculate_sigma_hessian(params)
        Sigma = opt.calculate_sigma(params)
        invSigma = np.linalg.pinv(Sigma)

        Cov = opt.m_cov
        n = len(params)
        hessian = np.zeros((n, n))
        for i, j in product(range(n), range(n)):
            dSi, dSj = sGrad[i], sGrad[j]
            ddS = sHess[i, j]
            t1 = invSigma @ ddS
            t2 = invSigma @ dSj @ invSigma
            t3 = invSigma @ dSi
            h = t1 - t2 @ dSi + Cov @ (t1 @ invSigma - t2 @ t3.T - t3 @ t2)
            hessian[i, j] = -np.trace(h)
        return hessian

    def calculate_information(params):
        Sigma = opt.calculate_sigma(params)
        sGrad = opt.calculate_sigma_gradient(params)
        invSigma = np.linalg.pinv(Sigma)

        # print(Sigma, sGrad, invSigma)

        sz = len(opt.params)
        I = np.zeros((sz, sz))
        for i, k in product(range(sz), range(sz)):
#             A = sGrad[k] @ invSigma @ sGrad[i]
#             I[i, k] = np.trace(invSigma @ ((1 - dof) * (sHess[i, k] - A) + dof * A.T))
             I[i, k] = np.trace(sGrad[i] @ invSigma @ sGrad[k] @ invSigma)
        return I
    if information == 'expected':
        information = calculate_information(opt.params)
    elif information == 'observed':
        information = calculate_hessian(opt.params)
        #information = calculate_hessian(opt.params)
    # print(information)
    asymptoticCov = np.linalg.pinv(information)
    variances = asymptoticCov.diagonal()
    return np.sqrt(variances / (data.m_profiles.shape[0] / 2))
        

def calculate_z_values(opt: SEMOptABC, data: SEMData, stdErrors=None):
    if stdErrors is None:
        stdErrors = calculate_standard_errors(opt, data)
    return [val / std for val, std in zip(list(opt.params), stdErrors)]


def calculate_p_values(opt: SEMOptABC, data: SEMData, zScores=None):
    if zScores is None:
        zScores = calculate_z_values(opt, data)
    return [2 * (1 - norm.cdf(abs(z))) for z in zScores]


def gather_statistics(opt: SEMOptABC, data: SEMData):
    values = opt.params.copy()
    stdErrors = calculate_standard_errors(opt, data)
    zScores = calculate_z_values(opt, data, stdErrors)
    pValues = calculate_p_values(opt, data, zScores)
    lh = calculate_likelihood(opt, data)
    aic = calculate_aic(opt, data, lh)
    bic = calculate_bic(opt, data, lh)
    paramStats = [ParameterStatistics(val, std, ztest, pvalue)
                  for val, std, ztest, pvalue
                  in zip(values, stdErrors, zScores, pValues)]
    dof = calculate_dof(opt, data)
    return SEMStatistics(dof, lh, aic, bic, paramStats)


def gather_pvals(opt: SEMOptABC, data: SEMData):
    values = opt.params.copy()
    stdErrors = calculate_standard_errors(opt, data)
    # print(stdErrors)
    zScores = calculate_z_values(opt, data, stdErrors)
    pValues = calculate_p_values(opt, data, zScores)
    lh = calculate_likelihood(opt, data)
    aic = calculate_aic(opt, data, lh)
    bic = calculate_bic(opt, data, lh)

    return pValues