from semopy_model import SEMData, SEMModel
import numpy as np
from scipy.optimize import minimize
from functools import partial
from scipy.stats import multivariate_normal, norm
import random


class SEMOptimiser:

    def __init__(self, mod: SEMModel, data: SEMData, estimator, regularizator=None):
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

        # Loss-functional and its additional parameters
        self.loss_func, self.add_params = self.get_loss_function(estimator)
        self.loss_func = partial(self.loss_func, self)
        self.add_param_bounds = [(None, None) for _ in range(len(self.add_params))]

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

    def get_loss_function(self, name):
        add_params = []  # additional parameters for optimisation
        if name == 'ULS':
            return SEMOptimiser.unweighted_least_sqares, add_params
        elif name == 'GLS':
            return SEMOptimiser.general_least_squares, add_params
        elif name == 'WLS':
            return SEMOptimiser.weighted_least_squares, add_params
        elif name == 'MLW':
            return SEMOptimiser.ml_wishart, add_params
        elif name == 'MLN':
            return SEMOptimiser.ml_normal, add_params
        elif name == 'Naive':
            return SEMOptimiser.naive_loss, add_params
        elif name == 'MLSkewed':
            # There are additional shape parameters of skewness in this method.
            # Number of parameters is a number of variables in m_cov
            add_params = np.ones(self.m_cov.shape[0]) * 0.05
            return SEMOptimiser.ml_skewed, add_params
        elif name == 'MLGamma':
            return SEMOptimiser.ml_gamma, add_params
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
            self.loss_func = partial(self.loss_func, alpha=0.01)
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
            cons = dict()
            if self.estimator == 'MLSkewed':
                self.loss_func = partial(self.loss_func, alpha=0.01)
                cons = ({'type': 'ineq', 'fun': lambda p: self.get_constr_skew_sigma(p)},
                        {'type': 'ineq', 'fun': lambda p: self.get_constr_skew_cov(p)})

            params_init = np.concatenate((self.params, self.add_params))
            loss = self.loss_func(params_init)

            # to save best parameters during minimisation
            self.min_loss = loss
            self.min_params = params_init
            res = minimize(self.loss_func, params_init,
                           constraints=cons,
                           method=optMethod, options=options,
                           bounds=self.param_bounds + self.add_param_bounds)
            if self.estimator != 'MLSkewed':
                self.params = res.x[0:len(self.params)]
                self.add_params = res.x[len(self.params):len(res.x)]
            else:
                self.params = self.min_params[0:len(self.params)]
                self.add_params = self.min_params[len(self.params):len(self.min_params)]

        params_out = np.concatenate((self.params, self.add_params))
        loss = (loss, self.loss_func(params_out))
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
        acc_log_exp1 = 0
        for y in m_profiles:
            acc_log_exp -= 1/2 * (log_det_sigma +
                                  y @ m_inv_sigma @ y +
                                  k*np.log(2*np.pi))
        return acc_log_exp


    def ml_normal(self, params, alpha=0.01):
        """
        Multivariate Normal Distribution
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

        # add regularization term
        loss = loss + self.regularizator(params) * alpha

        # Remember the best loss_func value
        if (loss < self.min_loss) and (loss > 0):
            self.min_loss = loss
            self.min_params = params

        return loss

    @staticmethod
    def ml_skw_log_cdf_ratio(m_sigma, skw, m_profiles):
        m_inv = np.linalg.inv(m_sigma)
        acc_log_val = 0
        for y in m_profiles:
            acc_log_val += np.log(norm.cdf((skw @ m_inv @ y) / np.sqrt(1 - skw @ m_inv @ skw)))
        return acc_log_val

    def ml_skewed(self, params, alpha=0.01):
        """
        Multivariate Skewed Normal Distribution
        :param params:
        :return:
        """
        print(len(params))
        # Divide parameters
        params_sem = params[0:len(self.params)]
        params_skw = params[len(self.params):len(params)]

        # print(len(params_sem), params_sem)
        # print(len(params_skw), params_skw)

        m_sigma = self.calculate_sigma(params_sem)
        m_cov = self.m_cov

        # TODO need to be removed: A kind of regularisation
        if self.get_constr_sigma(params_sem) < 0:
            return 10 ** 20
        if self.get_constr_skew_sigma(params) < 0:
            return 10 ** 20
        if self.get_constr_skew_cov(params) < 0:
            return 10 ** 20

        m_profiles = self.m_profiles

        # # ---------------
        # #   TMP
        # # ---------------
        # m_cov = sem_optimiser.m_cov
        # m_sigma = sem_optimiser.calculate_sigma(sem_optimiser.params)
        # m_profiles = sem_optimiser.m_profiles
        # var = multivariate_normal(np.zeros(m_cov.shape[0]), m_sigma)
        #
        # sem_optimiser.ml_norm_log_likelihood(m_cov, [m_profiles[0]])
        # sem_optimiser.ml_norm_log_likelihood(m_sigma, m_profiles)

        #---------------

        log_likelihood_sigma = self.ml_norm_log_likelihood(m_sigma, m_profiles)
        log_likelihood_cov = self.ml_norm_log_likelihood(m_cov, m_profiles)
        log_cdf_sigma = self.ml_skw_log_cdf_ratio(m_sigma, params_skw, m_profiles)
        log_cdf_cov = self.ml_skw_log_cdf_ratio(m_cov, params_skw, m_profiles)

        # if np.isnan(log_cdf_sigma)


        # loss = np.abs(log_likelihood_sigma - log_likelihood_cov +log_cdf_sigma - log_cdf_cov)
        loss = np.abs(log_likelihood_sigma - log_likelihood_cov + log_cdf_sigma)
        loss = np.abs(log_likelihood_sigma - log_likelihood_cov)
        loss = np.abs(log_likelihood_sigma)

        print(loss, log_likelihood_sigma, log_likelihood_cov, log_cdf_sigma, log_cdf_cov)

        # loss = self.ml_normal(params_sem, 0)
        if (loss < self.min_loss) and (loss > 0):
            self.min_loss = loss
            self.min_params = params
        # print(loss)
        return loss


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

    def get_constr_skew_sigma(self, params):
        params_sem = params[0:len(self.params)]
        params_skw = params[len(self.params):len(params)]

        # print(len(params_sem), params_sem)
        # print(len(params_skw), params_skw)

        m_sigma = self.calculate_sigma(params_sem)
        m_inv_sigma = np.linalg.pinv(m_sigma)
        return 1 - params_skw @ m_inv_sigma @ params_skw - 1e-6

    def get_constr_skew_cov(self, params):
        params_skw = params[len(self.params):len(params)]

        # print(len(params_sem), params_sem)
        # print(len(params_skw), params_skw)

        m_cov = self.m_cov
        m_inv = np.linalg.pinv(m_cov)
        return 1 - params_skw @ m_inv @ params_skw - 1e-6


    def gradient(self):
        def grad_coord(x):
            return (self.loss_func(self.params + x * eps) - self.loss_func(self.params))/eps
        eps = 1e-6
        g = np.array([grad_coord(x) for x in np.identity(len(self.params))])
        return g
