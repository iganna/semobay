from sem_model import SEMData, SEMModel
import numpy as np
from functools import partial
from abc import ABC, abstractmethod
from itertools import product
import scipy.stats as st


class SEMOptABC(ABC):

    def __init__(self, mod: SEMModel, data: SEMData):
        """
        Initialisation of the optimiser
        :param mod:
        :param data:
        :param estimator:
        """
        # TODO Does the model and the data are in agreement
        self.get_matrices = mod.get_matrices
        self.params = np.array(mod.param_val)
        self.initial_params = self.params.copy()
        self.param_pos = mod.param_pos
        self.param_bounds = mod.get_bounds()

        self.m_profiles = data.m_profiles
        self.m_cov = data.m_cov  # Covariance matrix

        n_prof = self.m_profiles.shape[0]
        n_stoch = round(n_prof * 0.9)
        self.n_cov_set = 100
        self.m_cov_set = \
            [np.cov(self.m_profiles[np.random.choice(n_prof,
                                                     n_stoch,
                                                     replace=False)],
                    rowvar=False,
                    bias=True)
             for _ in range(self.n_cov_set)]

        # for optimisation
        self.min_loss = 0
        self.min_params = self.params
        self.__prepare_diff_matrices(mod)

    @property
    def m_cov_stoch(self):
        # return self.m_cov_set[np.random.choice(self.n_cov_set, 1)[0]]
        n_prof = self.m_profiles.shape[0]
        n_stoch = round(n_prof * 0.99)
        return np.cov(self.m_profiles[np.random.choice(n_prof,
                                                       n_stoch,
                                                       replace=False)],
                      rowvar=False,
                      bias=True)

    def __prepare_diff_matrices(self, model: SEMModel):
        """Builds derivatives of each of matricies."""
        self.dParamsMatrices = list()
        ms = self.get_matrices()
        for k in range(model.n_param):
            mxType, i, j = model.param_pos[k]
            dMt = np.zeros_like(ms[mxType])
            dMt[i, j] = 1
            if mxType in {'Psi', 'Theta'}:
                dMt[j, i] = 1
            self.dParamsMatrices.append((mxType, dMt))

    @abstractmethod
    def loss_functions(self) -> dict:
        raise ValueError("Loss functions is not specified")

    def reset_params(self):
        self.params = self.initial_params.copy()

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
        Sigma matrix calculated from the model.
        """
        if params is None:
            params = self.params
        ms = self.get_matrices(params)
        Beta = ms['Beta']
        Lambda = ms['Lambda']
        Psi = ms['Psi']
        Theta = ms['Theta']
        
        C = np.linalg.pinv(np.identity(Beta.shape[0]) - Beta)
        M = Lambda @ C
        return M @ Psi @ M.T + Theta

    def calculate_sigma_gradient(self, params=None):
        if params is None:
            params = self.params
        ms = self.get_matrices(params)
        Beta = ms['Beta']
        Lambda = ms['Lambda']
        Psi = ms['Psi']
        C = np.linalg.pinv(np.identity(Beta.shape[0]) - Beta)
        M = Lambda @ C
        M_T = M.T
        K = C @ Psi
        KM_T = K @ M_T
        grad = list()
        for mxType, mx in self.dParamsMatrices:
            if mxType == 'Theta':
                grad.append(mx)
            elif mxType == 'Lambda':
                t = mx @ KM_T
                grad.append(t + t.T)
            elif mxType == 'Beta':
                t = mx @ K
                grad.append(M @ (t + t.T) @ M_T)
            elif mxType == 'Psi':
                grad.append(M @ mx @ M_T)
            else:
                grad.append(np.zeros_like(self.matrices['Theta']))
        return grad

    def calculate_sigma_hessian(self, params=None):
        if params is None:
            params = self.params
        ms = self.get_matrices(params)
        Beta = ms['Beta']
        Lambda = ms['Lambda']
        Psi = ms['Psi']
        zeroMatrix = np.zeros_like(ms['Theta'])
        n, m = len(params), zeroMatrix.shape[0]
        hessian = np.zeros((n, n, m, m))
        C = np.linalg.pinv(np.identity(Beta.shape[0]) - Beta)
        M = Lambda @ C
        M_T = M.T
        CPsi = C @ Psi
        CPsi_T = CPsi.T
        T = CPsi @ C.T
        for i, j in product(range(n), range(n)):
            aType, iMx = self.dParamsMatrices[i]
            bType, jMx = self.dParamsMatrices[j]
            if aType == 'Beta':
                if bType == 'Beta':
                    K = iMx @ CPsi
                    kSum = K + K.T
                    BiC = iMx @ C
                    BiC_T = BiC.T
                    BkC = jMx @ C
                    BkC_T = BkC.T
                    h = M @ (BkC @ kSum + kSum @ BkC_T + BiC @ BkC @ Psi +\
                             CPsi_T @ BkC_T @ BiC_T) @ M_T
                    hessian[i, j] = h
                elif bType == 'Lambda':
                    K = iMx @ CPsi
                    kSum = K + K.T
                    t = jMx @ C
                    hessian[i, j] = M @ kSum @ t.T + t @ kSum @ M_T
                elif bType == 'Psi':
                    K_hat = iMx @ C @ jMx
                    hessian[i, j] = M @ (K_hat + K_hat.T) @ M_T
                elif bType == 'Theta':
                    hessian[i, j] = zeroMatrix
            elif aType == 'Lambda':
                if bType == 'Beta':
                    K_hat = jMx @ CPsi
                    kSum = K_hat + K_hat.T
                    Mi = iMx @ C
                    hessian[i, j] = M @ kSum @ Mi.T + Mi @ kSum @ M_T
                elif bType == 'Lambda':
                    hessian[i, j] = iMx @ T @ jMx.T + jMx @ T @ iMx.T
                elif bType == 'Psi':
                    Mi = iMx @ C
                    hessian[i, j] = Mi @ jMx @ M_T + M @ jMx @ Mi.T
                elif bType == 'Theta':
                    hessian[i, j] = zeroMatrix
            elif aType == 'Psi':
                if bType == 'Beta':
                    K = jMx @ CPsi
                    kSum = K + K.T
                    hessian[i, j] = M @ kSum @ M_T
                elif bType == 'Lambda':
                    Mj = jMx @ C
                    hessian[i, j] = Mj @ iMx @ M_T + M @ iMx @ Mj.T
                else:
                    hessian[i, j] = zeroMatrix
            else:
                hessian[i, j] = zeroMatrix
        return hessian

    def ml_wishart(self, params):
        """
        F_wish = tr[S * Sigma^(-1)] + log(det(Sigma)) - log(det(S)) - (# of variables)
        We need to minimize the abs(F_wish) as it is a log of the ratio
        and the ration tends to be 1.
        :param params:
        :return:
        """
        Sigma = self.calculate_sigma(params)
        Cov = self.m_cov
        det_sigma = np.linalg.det(Sigma)
        det_cov = np.linalg.det(Cov)

        if det_sigma < 0:
            return 1000000
        log_det_ratio = np.log(det_sigma) - np.log(det_cov)

        inv_Sigma = np.linalg.pinv(Sigma)
        loss = np.trace(Cov @ inv_Sigma) + log_det_ratio - Cov.shape[0]
        return abs(loss)

    def ml_wishart_gradient(self, params):
        Sigma = self.calculate_sigma(params)
        Sigma_grad = self.calculate_sigma_gradient(params)
        # Cov = self.m_cov
        Cov = self.m_cov_stoch
        inv_Sigma = np.linalg.pinv(Sigma)
        cs = Cov @ inv_Sigma
        return np.array([np.trace(inv_Sigma @ g - cs @ g @ inv_Sigma)
                         for g in Sigma_grad])

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

    @staticmethod
    def ml_norm_log_likelihood_new(m_matrix, m_profiles):
        """

        :param m_matrix:
        :param m_profiles:
        :return:
        """
        acc_log_exp = 0
        for y in m_profiles:
            acc_log_exp += st.multivariate_normal.logpdf(x=y,
                                                         mean=y*0,
                                                         cov=m_matrix)

        return acc_log_exp


    def regu_l1(self, params):
        return np.sum((np.abs(params))) / len(params)

    def regu_l1_gradient(self, params):
        return np.sign(params) / len(params)

    def regu_l2(self, params):
        return np.linalg.norm(params) ** 2

    def regu_l2_gradient(self, params):
        return 2 * params

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
