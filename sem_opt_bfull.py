import numpy as np
from sem_model_full import SEMModelFull, SEMmx
from sem_model import SEMData
import itertools as it
from scipy.special import erfinv
import scipy.stats as st

class SEMOptBayesFull():
    def __init__(self, mod: SEMModelFull, data: SEMData):


        # -------------------------------------------
        # Get data profiles "d_" means "data"
        # -------------------------------------------
        self.d_g = mod.d_g
        self.d_u = mod.d_u
        self.d_v = mod.d_v

        # Proportion of samples of the same level
        # Cummulative fractions to calculate CDFs
        self.z_cumm_fractions = self.get_z_fractions()


        self.n_obs = self.d_g.shape[1]  # Number of observatioons (samples)
        self.n_eta = len(mod.d_vars['v_eta'])
        self.n_xi = len(mod.d_vars['v_xi'])
        self.n_omega = self.n_eta + self.n_xi
        self.n_g = len(mod.d_vars['v_g'])
        self.n_x = len(mod.d_vars['v_x'])
        self.n_z = len(mod.d_vars['v_v'])
        self.n_spart = self.n_eta + self.n_xi + self.n_g
        self.n_mpart = self.n_eta + self.n_xi + self.n_g

        self.get_matrix = mod.get_matrix

        # -------------------------------------------
        # Get parameters: initial values and annotation
        # -------------------------------------------
        # Set attributes for parameters
        self.param_val = np.array(mod.param_val)
        # It is a dictionary, that explains parameters:
        self.param_pos = mod.param_pos

        # -------------------------------------------
        # Parameters for prior distributions
        # -------------------------------------------
        # Phi matrices ~ Inverse Wishart:
        self.p_phi_xi_df, self.p_phi_xi_cov = self.get_params_phi_xi()
        self.p_phi_y_df, self.p_phi_y_cov = self.get_params_phi_y()
        # Calculate inverse matrices only one time:
        self.p_phi_xi_cov_inv = np.linalg.inv(self.p_phi_xi_cov)
        self.p_phi_y_cov_inv = np.linalg.inv(self.p_phi_y_cov)

        # Theta matrices ~ Inverse Gamma
        self.p_theta_delta_alpha, self.p_theta_delta_beta = \
            self.get_params_theta_delta()
        self.p_theta_eps_alpha, self.p_theta_eps_beta = \
            self.get_params_theta_eps()

        # Parameters for normal distribution of path coefficients in
        # Structural part
        self.p_spart_mean, self.p_spart_cov = self.get_params_spart()
        self.p_spart_cov_inv = np.linalg.inv(self.p_spart_cov)
        self.p_spart_loc = self.p_spart_cov_inv @ self.p_spart_mean
        self.p_spart_qform = \
            self.p_spart_mean.T @ self.p_spart_cov_inv @ self.p_spart_mean

        # Measurement part
        self.p_mpart_mean, self.p_mpart_cov = self.get_params_mpart()
        self.p_mpart_cov_inv = np.linalg.inv(self.p_mpart_cov)
        self.p_mpart_loc = self.p_mpart_cov_inv @ self.p_mpart_mean
        self.p_mpart_qform = \
            self.p_mpart_mean.T @ self.p_mpart_cov_inv @ self.p_mpart_mean



        # -------------------------------------------
        # Initial values for z-boundaries
        # -------------------------------------------
        # Percentiles: dictionary with keys as id of variable

          # Values

        # -------------------------------------------
        # Initial values for latent variables
        # -------------------------------------------
        self.z_bounds = self.calc_z_bounds()
        self.d_z = self.gibbs_z()
        self.d_y = self.gibbs_y()

        self.d_eta, self.d_xi = self.gibbs_omega()


        # -------------------------------------------
        # Combined data id defined as "getters":
        # m_omega = (m_eta; m_xi)
        # m_x = (m_u; m_v)
        # -------------------------------------------

    @property
    def d_omega(self):
        return np.concatenate((self.d_eta, self.d_xi), axis=0)

    @property
    def d_spart(self):
        return np.concatenate((self.d_eta, self.d_xi, self.d_y), axis=0)

    @property
    def d_mpart(self):
        return np.concatenate((self.d_eta, self.d_xi, self.d_y), axis=0)

    @property
    def d_x(self):
        return np.concatenate((self.d_u, self.d_z), axis=0)

    def optimise(self):

        params = np.array(self.param_val)
        mcmc = np.array(params)

        for _ in range(1000):

            # ---------------------------------
            # Samples all Parameters (errors first)
            # ---------------------------------
            # Structural Part
            self.gibbs_spart()
            # Measurement Part
            self.gibbs_mpart()

            # ---------------------------------
            # Sample all latent variables
            # ---------------------------------
            # First: For Binary (Genetic) and Ordinal
            self.z_bounds = self.calc_z_bounds()
            self.gibbs_z()
            self.gibbs_y()
            # Second: for pure latent variables
            self.gibbs_omega(params)

            mcmc = np.append(mcmc, self.param_val, axis=0)

    def get_params_phi_xi(self):
        """
        Get prior parameters for Exogenous latent variables:
        df and scale
        :return:
        """
        return 5, np.identity(self.n_xi)

    def get_params_phi_y(self):
        """
        Get prior parameters for latent variables correspond to genetic:
        df and scale
        :return:
        """
        return 5, np.identity(self.n_g)

    def get_params_theta_delta(self):
        """
        Get patameters for InvGamma distribution
        :return:
        """
        return np.ones(self.n_eta) * 9, np.ones(self.n_eta) * 4

    def get_params_theta_eps(self):
        """
        Get patameters for InvGamma distribution
        :return:
        """
        return np.ones(self.n_x) * 9, np.ones(self.n_eta) * 4

    def get_params_spart(self):
        """
        Get parameters of Normal Distribution for path coefficients in
        the Structural part
        :return:
        """
        return np.ones(self.n_eta, self.n_spart) * 0.8, \
               np.identity(self.n_spart)

    def get_params_mpart(self):
        """
        Get parameters of Normal Distribution for factor loadings
        in the Measurement part
        :return:
        """
        return np.ones(self.n_eta, self.n_mpart) * 0.8, \
               np.identity(self.n_mpart)

    def get_z_fractions(self):
        """
        This function computes levels of z and percentilles
        :return:
        # """
        # z_fractions = dict()
        # for i, v in enumerate(self.d_v):
        #     unique, counts = np.unique(v, return_counts=True)
        #     counts /= sum(counts)
        #
        #     z_fractions[i] = {u: c for u, c in zip(unique, counts)}
        #
        # # here is an example
        # return {0: [0, 0.5, 0.5, 1], 1:[0, 0.2, 0.3, 0.5, 1]}
        pass



    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------
    def gibbs_phi_xi(self):
        """

         Sampling covariance matrix Phi_xi
         parameters of Wishard distibution of Phi_xi
                  p_phi_xi_df - degrees of freedom,
                  p_phi_xi_cov - matrix
          :return matrix Phi_xi:
         """
        d_xi = self.d_xi  # This is already sampled by the Gibbs sampling
        n_samples = self.n_obs

        # Posterior parameters
        # new p_phi_xi_df
        p_xi_cov_port = np.linalg.pinv(self.p_phi_xi_cov_inv + d_xi @ d_xi.T)
        # new p_phi_xi_df
        p_xi_df_post = self.p_phi_xi_df + n_samples

        m_phi_xi = st.invwishart.rvs(p_xi_df_post, p_xi_cov_port)

        for mx_type, pos1, pos2, param_id in self.param_pos:
            if mx_type is SEMmx.PHI_XI:
                self.param_val[param_id] = m_phi_xi[pos1, pos2]

    def gibbs_phi_y(self):
        """
        Gibbs sampling of Phi_y
           Sampling covariance matrix Phi_y
         :return matrix Phi_y,
                parameters of Wishard distibution of Phi_y:
                p_phi_y_df - degrees of freedom,
                 p_phi_y_cov - matrix """

        d_y = self.d_y  # This is already sampled by the Gibbs sampling
        n_samples = self.n_obs

        # Posterior parameters
        # new p_phi_y_df
        p_y_cov_port = np.linalg.pinv(self.p_phi_y_cov_inv + d_y @ d_y.T)
        # new p_phi_y_df
        p_y_df_post = self.p_phi_y_df + n_samples

        m_phi_y = st.invwishart.rvs(p_y_cov_port, p_y_df_post)

        for mx_type, pos1, pos2, param_id in self.param_pos:
            if mx_type is SEMmx.PHI_Y:
                self.param_val[param_id] = m_phi_y[pos1, pos2]

    def calc_z_bounds(self):
        """
        This function calculates alpha-values for boundaries
        :return:
        """
        m_sigma_z = self.get_matrix(SEMmx.SIGMA_Z, self.param_val)
        percentiles = dict()
        for i in range(self.n_z):
            variance = m_sigma_z[i][i]
            percentiles[i] = st.norm.ppf(self.z_cumm_fractions[i],
                                         scale=variance**(1/2))

        return percentiles

    def gibbs_z(self):
        """
        Gibbs sampling of Z variables
        :param params:
        :return:
        """
        d_z = np.zeros((self.n_x, self.n_obs))
        m_sigma_z = self.get_matrix(SEMmx.SIGMA_Z, self.param_val)

        for i in range(self.n_obs):
            z_tmp = \
                st.multivariate_normal(mean=np.zeros(self.n_g),
                                       cov=m_sigma_z)
            d_z[:, i] = z_tmp

        # Chack for correct class
        for i, j in it.product(range(self.n_z), range(self.n_obs)):

            value_norm = d_z[i, j]
            value_ord = self.d_v[i, j]

            if not((value_norm < self.z_bounds[i][value_ord]) and
                   (value_norm > self.z_bounds[i][value_ord-1])):
                d_z[i, j] = 0

        self.d_z = d_z

    def gibbs_y(self):
        """
        Gibbs sampling of Y variables
        Sampling Y -- latent variables of genotypes
        :return Y sample as matrix """

        m_phi_y = self.get_matrix(SEMmx.PHI_Y, self.param_val)
        d_y = np.zeros((self.n_g, self.n_obs))
        for i in range(self.n_obs):
            y_tmp = \
                st.multivariate_normal(mean=np.zeros(self.n_g),
                                       cov=m_phi_y)
            y_new = [y if (g == 1) == (y > 0) else 0
                     for y, g in zip(y_tmp, self.d_g[:, i])]
            d_y[:, i] = y_new
        return d_y

    def gibbs_omega(self, params=None):
        """
        Sampling Omega -- latent variables
        result: new sample Omega"""
        m_inv_sigma_x = np.linalg.pinv(self.get_matrix(SEMmx.SIGMA_Z,
                                                       params))
        m_ins_sigma_omega = np.linalg.pinv(self.get_matrix(SEMmx.SIGMA_OMEGA,
                                                           params))
        m_lambda = self.get_matrix(SEMmx.LAMBDA, params)
        m_kappa = self.get_matrix(SEMmx.KAPPA, params)
        m_inv_q = m_lambda.T @ m_inv_sigma_x @ m_lambda + m_ins_sigma_omega
        m_q = np.linalg.pinv(m_inv_q)

        # X  матрица, столбцы X - элементы выборки
        d_omega = np.zeros((20, self.n_obs))
        for i in range(self.n_obs):
            x = self.d_x[:, i]
            y = self.d_y[:, i]
            q = m_lambda.T @ m_inv_sigma_x @ (x - m_kappa @ y)
            d_omega[:, i] = np.random.normal(m_q @ q, m_q)

        d_eta = d_omega[0:self.n_eta, :]
        d_xi = d_omega[self.n_eta:, :]
        return d_eta, d_xi

    def gibbs_spart(self):
        """
         Sampling covariance matrixes Theta_delta, B, Pi, Gamma
          :return matrix Theta_delta,
                  parameter of Gamma distribution of Theta_delta:
                   p_theta_delta_alpha,
                  matrixes B, Pi, Gamma,
                  parameters of Normal distribution of matrix(B, Pi, Gamma):
                  p_b_pi_gamma_means, p_b_pi_gamma_covs """

        n_obs = self.n_obs
        d_spart = self.d_spart
        d_eta = self.d_eta

        # Calculate auxiliary variables
        a_cov_inv = self.p_spart_cov_inv + d_spart @ d_spart.T
        a_cov = np.linalg.inv(a_cov_inv)
        a_mean = a_cov @ (self.p_spart_loc + d_spart @ d_eta.T)

        # Calculate new parameters of InvGamma dna InvWishart
        p_alpha = self.p_theta_delta_alpha + n_obs / 2
        p_beta = self.p_theta_delta_beta + 1/2 * \
                 (d_eta.T @ d_eta - a_mean.T @ a_cov_inv @ a_mean +
                  self.p_spart_qform)

        # Sampling Theta and (Beta, Gamma, Pi)
        theta_diag = np.zeros(self.n_eta)
        path_coefs = np.zeros((self.n_eta, self.n_spart))
        for i in range(self.n_eta):
            theta_diag[i] = st.invgamma.rvs(a=p_alpha[i],
                                            scale=p_beta[i])
            path_coefs[i, :] = \
                st.multivariate_normal.rvs(mean=a_mean,
                                           cov=theta_diag[i]*a_cov)

        # Parse combined matrix into sub-matrices
        m_beta = path_coefs[:, 1:self.n_eta]
        m_gamma = path_coefs[:, self.n_eta:]
        m_pi = path_coefs[:, (self.n_eta + self.n_xi):]

        # Save corresponding parameters to the vector of parameters
        for mx_type, pos1, pos2, param_id in self.param_pos:
            if mx_type is SEMmx.THETA_DELTA:
                self.param_val[param_id] = theta_diag[pos1]

            if mx_type is SEMmx.BETA:
                self.param_val[param_id] = m_beta[pos1, pos2]

            if mx_type is SEMmx.GAMMA:
                self.param_val[param_id] = m_gamma[pos1, pos2]

            if mx_type is SEMmx.PI:
                self.param_val[param_id] = m_pi[pos1, pos2]

    def gibbs_mpart(self):
        """
         Sampling covariance matrixes Theta_delta, B, Pi, Gamma
          :return matrix Theta_delta,
                  parameter of Gamma distribution of Theta_delta:
                   p_theta_delta_alpha,
                  matrixes B, Pi, Gamma,
                  parameters of Normal distribution of matrix(B, Pi, Gamma):
                  p_b_pi_gamma_means, p_b_pi_gamma_covs """

        n_obs = self.n_obs
        d_mpart = self.d_mpart
        d_x = self.d_x

        # Calculate auxiliary variables
        a_cov_inv = self.p_mpart_cov_inv + d_mpart @ d_mpart.T
        a_cov = np.linalg.inv(a_cov_inv)
        a_mean = a_cov @ (self.p_mpart_loc + d_mpart @ d_x.T)

        # Calculate new parameters of InvGamma dna InvWishart
        p_alpha = self.p_theta_eps_alpha + n_obs / 2
        p_beta = self.p_theta_eps_beta + 1/2 * \
                 (d_x.T @ d_x - a_mean.T @ a_cov_inv @ a_mean +
                  self.p_mpart_qform)

        # Sampling Theta and (Beta, Gamma, Pi)
        theta_diag = np.zeros(self.n_eta)
        path_coefs = np.zeros((self.n_eta, self.n_mpart))
        for i in range(self.n_eta):
            theta_diag[i] = st.invgamma.rvs(a=p_alpha[i],
                                            scale=p_beta[i])
            path_coefs[i, :] = \
                st.multivariate_normal.rvs(mean=a_mean,
                                           cov=theta_diag[i]*a_cov)

        # Parse combined matrix into sub-matrices
        m_lambda = path_coefs[:, 1:self.n_omega]
        m_kappa = path_coefs[:, self.n_omega:]

        # Save corresponding parameters to the vector of parameters
        for mx_type, pos1, pos2, param_id in self.param_pos:
            if mx_type is SEMmx.THETA_EPS:
                self.param_val[param_id] = theta_diag[pos1]

            if mx_type is SEMmx.LAMBDA:
                self.param_val[param_id] = m_lambda[pos1, pos2]

            if mx_type is SEMmx.KAPPA:
                self.param_val[param_id] = m_kappa[pos1, pos2]

