from sem_model import SEMData, SEMModel
import numpy as np
from scipy.optimize import minimize
from functools import partial
from scipy.stats import multivariate_normal, norm
import random
from sem_opt_abc import SEMOptABC


class SEMOptSkewed(SEMOptABC):

    def __init__(self, mod: SEMModel, data: SEMData, estimator, regularization=None):
        """

        :param mod:
        :param data:
        :param estimator:
        :param regularizator:
        """
        super().__init__(mod, data, estimator, regularization)
        # Loss-functional and its additional parameters
        self.loss_func = self.get_loss_function(estimator)
        # New parameters and bound for new parameters
        self.add_params = np.ones(self.m_cov.shape[0]) * 0.05
        self.add_param_bounds = [(None, None) for _ in range(len(self.add_params))]


    @staticmethod
    def loss_functions():
        """
        Create the dictionary of possible functions
        :return:
        """
        tmp_dict = dict()
        tmp_dict['MLSkewed'] = SEMOptSkewed.ml_skewed
        tmp_dict['MLGamma'] = SEMOptSkewed.ml_gamma

        return tmp_dict


    def optimize(self, optMethod='SLSQP', bounds=None, alpha=0):
        """
           sigma - an empirical covariance matrix, lossFunction - a name of loss function to be
           used, optMethod - a scipy optimization method,
           *args - extra parameters for the lossFunction
        :param optMethod:
        :param bounds:
        :return:
        """


        options = {'maxiter': 1e3}


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



    def ml_normal(self, params):
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
