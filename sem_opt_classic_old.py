from sem_model import SEMData, SEMModel
import numpy as np
from scipy.optimize import minimize
from functools import partial
from scipy.stats import multivariate_normal, norm
import random
from sem_opt_abc import SEMOptABC


class SEMOptClassic(SEMOptABC):

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



    @staticmethod
    def loss_functions() -> dict:
        """
        Create the dictionary of possible functions
        :return:
        """
        tmp_dict = dict()
        tmp_dict['ULS'] = SEMOptClassic.unweighted_least_squares
        tmp_dict['GLS'] = SEMOptClassic.general_least_squares
        tmp_dict['WLS'] = SEMOptClassic.general_least_squares
        tmp_dict['MLW'] = SEMOptClassic.ml_wishart
        tmp_dict['MLN'] = SEMOptClassic.ml_normal
        tmp_dict['Naive'] = SEMOptClassic.naive_loss
        return tmp_dict



    def optimize(self, opt_method='SLSQP', bounds=None, alpha=0):
        """

        :param optMethod:
        :param bounds:
        :param alpha:
        :return:
        """
        def func_to_min(params):
            """ Sum of loss function and regularisation """
            p_beta = [p for i, p in enumerate(params) if self.param_pos[i][0]
                      == 'Beta']
            return self.loss_func(self, params) + \
                   alpha * self.regularization(p_beta)

        # Specify initial parameters and function to minimize
        params_init = self.params

        # Minimisation
        options = {'maxiter': 1e3}

        cons = ({'type': 'ineq', 'fun': lambda p: self.constraint_all(p)})

        res = minimize(func_to_min, params_init,
                       constraints=cons,
                       method=opt_method, options=options,
                       bounds=self.param_bounds)

        # Save results
        self.params = res.x

        return func_to_min(params_init), func_to_min(self.params)

    def unweighted_least_squares(self, params):
        m_sigma = self.calculate_sigma(params)
        m_cov = self.m_cov
        t = m_sigma - m_cov
        loss = np.trace(np.matmul(t, t.T))
        print(loss)
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
        t = (m_cov - m_sigma) @ w
        loss = np.trace(np.matmul(t, t.T))
        return loss

    def weighted_least_squares(self, params, weights):
        m_sigma = self.calculate_sigma(params)
        m_cov = self.m_cov
        t = m_sigma - m_cov
        w = np.linalg.inv(weights.T * weights)
        return np.trace(np.matmul(np.matmul(t, w), t.T))


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
        if self.constraint_sigma(params) < 0:
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


