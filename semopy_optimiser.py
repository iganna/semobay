from semopy_model import SEMData, SEMModel
import numpy as np
from scipy.optimize import minimize
from functools import partial
from scipy.stats import multivariate_normal
import random


class SEMOptimiser:

    def __init__(self, mod: SEMModel, data: SEMData, estimator, regularizator = None):
        """
        Initialisation of the optimiser
        :param mod:
        :param data:
        :param estimator:
        """
        # TODO Does the model and the data are in agreement
        self.regularizator = self.get_regularization(regularizator)
        self.estimator = estimator
        self.get_matrices = mod.get_matrices
        self.params = mod.param_val
        self.param_bounds = mod.get_bounds()

        self.m_profiles = data.m_profiles
        self.m_cov = data.m_cov  # Covariance matrix
        self.loss_func = partial(self.get_loss_function(estimator), self)

        # for optimisation
        self.min_loss = 0
        self.min_params = self.params

    def calculate_sigma(self, params=None):
        """
        Sigma matrix calculated from the model
        """
        if params is None:
            params = self.params
        ms = self.get_matrices(params)
        m_beta = ms['Beta']
        m_lambda = ms['Lambda']
        m_psi = ms['Psi']
        m_theta = ms['Theta']

        m_c = np.linalg.pinv(np.identity(m_beta.shape[0]) - m_beta)
        return m_lambda @ m_c @ m_psi @ m_c.T @ m_lambda.T + m_theta

    @staticmethod
    def get_loss_function(name):
        if name == 'ULS':
            return SEMOptimiser.unweighted_least_sqares
        elif name == 'GLS':
            return SEMOptimiser.general_least_squares
        elif name == 'WLS':
            return SEMOptimiser.weighted_least_squares
        elif name == 'MLW':
            return SEMOptimiser.ml_wishart
        elif name == 'MLN':
            return SEMOptimiser.ml_normal
        elif name == 'Naive':
            return SEMOptimiser.naive_loss
        elif name == 'MLSkewed':
            return SEMOptimiser.ml_skewed
        elif name == 'MLGamma':
            return SEMOptimiser.ml_gamma
        else:
            raise Exception("ScipyBackend doesn't support loss function {}.".format(name))

    @staticmethod
    def get_regularization(name):
        if name == 'L1':
            return SEMOptimiser.regul_l1
        elif name == 'L2':
            return SEMOptimiser.regul_l1
        else:
            return SEMOptimiser.regul_zero

    def optimize(self, optMethod='SLSQP', bounds=None):
        """
           sigma - an empirical covariance matrix, lossFunction - a name of loss function to be
           used, optMethod - a scipy optimization method,
           *args - extra parameters for the lossFunction
        :param optMethod:
        :param bounds:
        :return:
        """


        options = {'maxiter': 1e3}


        if self.estimator == 'MLN':
            # options['ftol'] = 0.000001
            options['disp'] = True
            self.loss_func = partial(self.loss_func, alpha=0.1)
            loss = self.loss_func(self.params)

            # to save best parameters during minimisation
            self.min_loss = loss
            self.min_params = self.params

            cons = ({'type': 'ineq', 'fun': lambda p: self.get_constr_theta(p)},
                    {'type': 'ineq', 'fun': lambda p: self.get_constr_psi(p)},
                    {'type': 'ineq', 'fun': lambda p: self.get_constr_sigma(p)})
            res = minimize(self.loss_func, self.params,
                           constraints=cons,
                           method='SLSQP', options=options,
                           bounds=self.param_bounds)
            # TODO: another strange moment
            # self.params = res.x
            self.params = self.min_params

        else:
            loss = self.loss_func(self.params)
            res = minimize(self.loss_func, self.params,
                           method=optMethod, options=options,
                           bounds=self.param_bounds)
            self.params = res.x



        loss = (loss, self.loss_func(self.params))
        return loss

    def unweighted_least_sqares(self, params):
        m_sigma = self.calculate_sigma(params)
        m_cov = self.m_cov
        t = m_sigma - m_cov
        loss = np.trace(np.matmul(t, t.T))
        # print(loss, self.general_least_squares(params))
        return loss

    def naive_loss(self, params):
        m_sigma = self.calculate_sigma(params)
        m_cov = self.m_cov
        t = m_cov - m_sigma
        return np.linalg.norm(t)

    def general_least_squares(self, params):
        m_sigma = self.calculate_sigma(params)
        m_cov = self.m_cov
        w = np.linalg.inv(m_cov)
        # w = np.identity(m_cov.shape[0])
        t = (m_cov - m_sigma) @ w
        loss = np.trace(np.matmul(t, t.T))

        print(loss, self.unweighted_least_sqares(params))
        return loss

    def weighted_least_squares(self, params, weights):
        m_sigma = self.calculate_sigma(params)
        m_cov = self.m_cov
        t = m_sigma - m_cov
        w = np.linalg.inv(weights.T * weights)
        return np.trace(np.matmul(np.matmul(t, w), t.T))

    def ml_wishart(self, params):
        """
        F_wish = tr[S * Sigma^(-1)] + log(det(Sigma)) - log(det(S)) - (# of variables)
        We need to minimize the abs(F_wish) as it is a log of the ratio
        and the ration tends to be 1.
        :param params:
        :return:
        """
        m_sigma = self.calculate_sigma(params)
        m_cov = self.m_cov
        det_ratio = np.linalg.det(m_sigma) / np.linalg.det(m_cov)
        # 1e6 stands for "infinity" here. I've tried a lot of numerical shenanigans
        # so far, of little avail though.
        log_det_ratio = np.log(det_ratio + 1e-16) if det_ratio > 0 else 1e6
        m_inv_sigma = np.linalg.pinv(m_sigma)
        loss = np.trace(np.matmul(m_cov, m_inv_sigma)) + log_det_ratio - m_cov.shape[0]

        return abs(loss)

    @staticmethod
    def ml_norm_log_likelihood(m_sigma, m_profiles):
        det_sigma = np.linalg.det(m_sigma)
        log_det_sigma = np.log(det_sigma)
        m_inv_sigma = np.linalg.inv(m_sigma)
        k = m_sigma.shape[0]
        acc_log_exp = 0
        for y in m_profiles:
            acc_log_exp -= 1/2 * (log_det_sigma +
                                  y @ m_inv_sigma @ y +
                                  k*np.log(2*np.pi))
        return acc_log_exp


    def ml_normal(self, params, alpha=0.01):
        """

        :param params:
        :param alpha:
        :return:
        """

        m_sigma = self.calculate_sigma(params)
        m_cov = self.m_cov

        # TODO need to be removed: A kind of regularisation
        if self.get_constr_sigma(params) < 0:
            return 10 ** 20

        m_profiles = self.m_profiles
        log_likelihood_sigma = self.ml_norm_log_likelihood(m_sigma, m_profiles)
        log_likelihood_cov = self.ml_norm_log_likelihood(m_cov, m_profiles)
        loss = - (log_likelihood_sigma - log_likelihood_cov)

        # TODO: Strange moment
        if loss < 0:
            return self.min_loss
        print(loss)

        # add regularization term
        loss = loss + self.regularizator(params) * alpha

        # Remember the best loss_func value
        if (loss < self.min_loss) and (loss > 0):
            self.min_loss = loss
            self.min_params = params



        return loss


    def ml_skewed(self, params):
        pass

    def ml_gamma(self, params):
        pass

    @staticmethod
    def regul_l1(params):
        return sum(np.abs(params)) / len(params)

    @staticmethod
    def regul_l2(params):
        return np.linalg.norm(params)

    @staticmethod
    def regul_zero(params):
        return 0

    def get_constr_theta(self, params):
        ms = self.get_matrices(params)
        return np.linalg.det(ms['Theta']) - 1e-6

    def get_constr_psi(self, params):
        ms = self.get_matrices(params)
        return np.linalg.det(ms['Psi']) - 1e-6

    def get_constr_sigma(self, params):
        m_sigma = self.calculate_sigma(params)
        # return np.linalg.det(m_sigma) - 1e-6
        return sum(np.linalg.eig(m_sigma)[0]>0) - m_sigma.shape[0]


    def gradient(self):
        def grad_coord(x):
            return (self.loss_func(self.params + x * eps) - self.loss_func(self.params))/eps
        eps = 1e-6
        g = np.array([grad_coord(x) for x in np.identity(len(self.params))])
        return g
