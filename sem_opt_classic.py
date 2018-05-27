from sem_model import SEMData, SEMModel
import numpy as np
from scipy.optimize import minimize
from functools import partial
from scipy.stats import multivariate_normal, norm
import random
from sem_opt_abc import SEMOptABC
from math import log


class SEMOptClassic(SEMOptABC):

    def __init__(self, mod: SEMModel, data: SEMData, estimator, regularization=None):
        """
        :param mod:
        :param data:
        :param estimator:
        """

        super().__init__(mod, data)
        # Loss-functional and its additional parameter

        self.loss_func = self.get_loss_function(estimator)
        self.estimator = estimator
        self.regularization = regularization

        self.param_fixed = []
        self.param_zeros = []

    def fix_matrix(self, mx_type):
        self.param_fixed += [k for k, i in self.param_pos.items()
                             if i[0] in mx_type]

        self.param_fixed = list(set(self.param_fixed))

    def fix_param_zero(self, param_id):
        self.params[param_id] = 0
        self.param_fixed += [param_id]
        self.param_fixed = list(set(self.param_fixed))



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
        
    def get_obj_function(self,  name: str):
        objDict = {'ULS': self.unweighted_least_squares,
                   'MLW': self.ml_wishart,
                   'MLN': self.ml_normal,
                   'GLS': self.general_least_squares}
        try:
            return objDict[name]
        except KeyError:
            raise Exception("Unknown optimization method {}.".format(name))

    def get_gradient_function(self, name: str):
        gradDict = {'MLW': self.ml_wishart_gradient,
#                    'ULS': self.uls_gradient,
#                    'GLS': self.gls_gradient,
                    'l1':  self.regu_l1_gradient,
                    'l2':  self.regu_l2_gradient}
        if name in gradDict:
            return gradDict[name]
        else:
            return None

    def get_regularization(self, name: str):
        reg_dict = {'l1': self.regu_l1,
                    'l2': self.regu_l2}
        return reg_dict[name]

    def compose_loss_function(self, alpha=0):
        """Build a loss function.
        Key arguments:
        method -- a name of an optimization technique to apply.
        regularization -- a name of regularizatio technique to apply.
        paramsToPenalize -- indicies of parameters from params' vector to 
                            penalize. If None, then all params are penalized.
        a - a regularization multiplier.
        Returns:
        (Loss function, obj_func, a * regularization)
        Loss function = obj_func + a * regularization"""

        regularization = self.regularization

        obj_func = self.get_obj_function(self.estimator)
        reg_func = None
        if regularization is not None:
            reg_func = self.get_regularization(regularization)
            loss_func = lambda params: obj_func(params) + \
                                       alpha * reg_func(params)
        else:
            loss_func = lambda params: obj_func(params)
        grad_func = self.compose_gradient_function(alpha=alpha)
        return loss_func, obj_func, reg_func, grad_func

    def compose_gradient_function(self, alpha=0):
        """ Builds a gradient function if possible. """

        def grad_composed(params):
            g = grad_of(params)
            rg = alpha * regularization(params)
            res = g + rg
            res = [0 if i in self.param_fixed else g
                   for i, g in enumerate(res)]
            return res

        # def grad_zero(grad, params):
        #     g = grad(params)
        #     print('anna')
        #
        #     return res


        regularization = self.regularization
        grad = None
        grad_of = self.get_gradient_function(self.estimator)
        if grad_of is not None:
            grad = grad_of
        if regularization is not None and grad is not None:
            regularization = self.get_gradient_function(regularization)
            if regularization is None:
                return None
            grad = grad_composed

        # grad = grad_zero
        return grad

    def optimize(self, opt_method='SLSQP', bounds=None, alpha=0):
        """
        :param opt_method:
        :param bounds:
        :param alpha:
        :return:
        """



        # Minimisation
        options = {'maxiter': 10000, 'disp': False}

        loss_func, obj_func, reg_func, grad_func = \
            self.compose_loss_function(alpha=alpha)
        if grad_func is None:
            print("Warning: analytical gradient is not available.")

        # Specify initial parameters and function to minimize
        params_prev = np.array(self.params)
        params_init = np.array(self.params)
        params_first = np.array(self.params)
        loss_prev = loss_func(params_init)
        loss_init = loss_func(params_init)


        n_no = 0
        for _ in range(10):

            res = minimize(loss_func, params_init,
                           jac=grad_func,
                           method=opt_method, options=options,
                           bounds=self.param_bounds)
            if res.fun > loss_init:
                thresh = 1 - (res.fun - loss_init)
                if thresh < 0:
                    n_no += 1

                    if n_no > 2:
                        params_init = params_prev
                        loss_init = loss_prev
                        n_no = 0

                    continue

                if np.random.rand(1) <= thresh:
                    continue
            params_prev = params_init
            loss_prev = loss_init
            params_init = res.x
            loss_init = res.fun
            n_no = 0

        # Save results
        self.params = np.array(params_init)

        print(loss_func(params_init), loss_func(params_first))

        return loss_func(params_init), loss_func(params_first)

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


