from sem_model import SEMData, SEMModel
import numpy as np
from functools import partial
from abc import ABC, abstractmethod
from sem_regul import get_regul

class SEMOptABC(ABC):

    def __init__(self, mod: SEMModel, data: SEMData, estimator, regularization=None):
        """
        Initialisation of the optimiser
        :param mod:
        :param data:
        :param estimator:
        """
        # TODO Does the model and the data are in agreement
        self.regularization = get_regul(regularization)
        self.estimator = estimator
        self.get_matrices = mod.get_matrices
        self.params = np.array(mod.param_val)
        self.param_pos = mod.param_pos
        self.param_bounds = mod.get_bounds()

        self.m_profiles = data.m_profiles
        self.m_cov = data.m_cov  # Covariance matrix

        # for optimisation
        self.min_loss = 0
        self.min_params = self.params

    @abstractmethod
    def loss_functions(self) -> dict:
        raise ValueError("Loss functions is not specified")

    @abstractmethod
    def optimize(self, opt_method='SLSQP', bounds=None, alpha=0):
        raise ValueError("Optimizer is not specified")

    def get_loss_function(self, name):
        loss_dict = self.loss_functions()
        if name in loss_dict.keys():
            return loss_dict[name]
        else:
            raise Exception("SEMopy Backend doesn't support loss function {}.".format(name))


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
    def ml_norm_log_likelihood(m_matrix, m_profiles):
        det_sigma = np.linalg.det(m_matrix)
        log_det_sigma = np.log(det_sigma)
        m_inv_sigma = np.linalg.inv(m_matrix)
        k = m_matrix.shape[0]
        acc_log_exp = 0
        acc_log_exp1 = 0
        for y in m_profiles:
            acc_log_exp -= 1/2 * (log_det_sigma +
                                  y @ m_inv_sigma @ y +
                                  k*np.log(2*np.pi))
        return acc_log_exp

    def constraint_theta(self, params):
        ms = self.get_matrices(params)
        # return np.linalg.det(ms['Theta']) - 1e-6
        # return sum(np.linalg.eig(ms['Theta'])[0] > 0) - ms['Theta'].shape[0]
        return sum(ms['Theta'].diagonal() >= 0) - ms['Theta'].shape[0]
    def constraint_psi(self, params):
        ms = self.get_matrices(params)
        # return np.linalg.det(ms['Psi']) - 1e-6
        return sum(np.linalg.eig(ms['Psi'])[0] > 0) - ms['Psi'].shape[0]

    def constraint_sigma(self, params):
        m_sigma = self.calculate_sigma(params)
        # return np.linalg.det(m_sigma) - 1e-6
        return sum(np.linalg.eig(m_sigma)[0] > 0) - m_sigma.shape[0]

    def constraint_all(self, params):
        return self.constraint_psi(params) + \
               self.constraint_sigma(params) + \
               self.constraint_theta(params)

    #
    # def gradient(self):
    #     def grad_coord(x):
    #         return (self.loss_func(self.params + x * eps) - self.loss_func(self.params))/eps
    #     eps = 1e-6
    #     g = np.array([grad_coord(x) for x in np.identity(len(self.params))])
    #     return g
