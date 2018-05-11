import numpy as np
from sem_model_full import SEMModelFull, SEMmx
from scipy.stats import invwishart, invgamma, wishart, norm, uniform, multivariate_normal
from scipy.special import erfinv
import scipy.stats as st

class SEMOptBayesFull():
    def __init__(self, mod: SEMModelFull, data: SEMData):

        self.get_matrix = mod.get_matrix
        self.n_obs = 0  # Number of observatioons (samples)
        self.n_eta = 0
        self.n_xi = 0
        self.n_g = 0

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
        self.p_phi_xi_cov_inv = np.linalg.pinv(self.p_phi_xi_cov)
        self.p_phi_y_cov_inv = np.linalg.pinv(self.p_phi_y_cov)

        # Theta matrices ~ Inverse Gamma
        self.p_theta_delta_alpha, self.p_theta_delta_beta = \
            self.get_params_theta_delta()
        self.p_theta_eps_alpha, self.p_theta_eps_beta = \
            self.get_params_theta_eps()

        # Parameters for normal distribution of path coefficients in
        # Structural part
        self.p_spart_mean, self.p_spart_cov = self.get_params_spart()
        # Measurement part
        self.p_mpart_mean, self.p_mpart_cov = self.get_params_mpart()

        # -------------------------------------------
        # Get data profiles "d_" means "data"
        # -------------------------------------------
        self.d_g = 0
        self.d_u = 0
        self.d_v = 0

        self.z_percent = self.get_z_percent()

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

    def optimise(self):

        params = np.array(self.param_val)
        mcmc = np.array(params)

        for _ in range(1000):
            # ---------------------------------
            # Samples all Parameters (errors first)
            # ---------------------------------
            # Structural Part
            params = self.gibbs_theta_delta(params)
            params = self.gibbs_spart(params)
            # Measurement Part
            params = self.gibbs_theta_eps(params)
            params = self.gibbs_mpart(params)

            # ---------------------------------
            # Sample all latent variables
            # ---------------------------------
            # First: For Binary (Genetic) and Ordinal
            self.z_bounds = self.calc_z_bounds(params)
            self.d_z = self.gibbs_z(params)
            self.d_y = self.gibbs_y(params)
            # Second: for pure latent variables
            self.d_eta, self.d_xi = self.gibbs_omega(params)

            mcmc = np.append(mcmc, params, axis=0)

    def d_omega(self):
        return np.concatenate((self.d_eta, self.d_xi), axis=0)

    def d_x(self):
        return np.concatenate((self.d_u, self.d_z), axis=0)

    def get_params_phi_xi(self):
        """
        Get prior parameters for Exogenous latent variables:
        df and scale
        :return:
        """
        return 0,0

    def get_params_phi_y(self):
        """
        Get prior parameters for latent variables correspond to genetic:
        df and scale
        :return:
        """
        return 0, 0

    def get_params_theta_delta(self):
        """
        Get patameters for InvGamma distribution
        :return:
        """
        return 0,0

    def get_params_theta_eps(self):
        """
        Get patameters for InvGamma distribution
        :return:
        """
        return 0,0

    def get_params_spart(self):
        """
        Get parameters of Normal Distribution for path coefficients in
        the Structural part
        :return:
        """
        return 0,0

    def get_params_mpart(self):
        """
        Get parameters of Normal Distribution for factor loadings
        in the Measurement part
        :return:
        """
        return 0,0

    def get_z_percent(self):
        """
        This function computes levels of z and percentilles
        :return:
        """
        # Use sigma Z
        sigma_z = self.get_matrix(self.param_val, SEMmx.SIGMA_Z)
        # here is an example
        return {0: [0.5, 0.5], 1:[0.2, 0.3, 0.5]}

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------
    def gibbs_psi_xi(self):
        """

         Sampling covariance matrix Phi_xi
         parameters of Wishard distibution of Phi_xi
                  p_phi_xi_df - degrees of freedom,
                  p_phi_xi_cov - matrix
          :return matrix Psi_xi:
         """
        d_xi = self.d_xi  # This is already sampled by the Gibbs sampling
        n_samples = self.n_obs

        # Posterior parameters
        # new p_phi_xi_df
        p_xi_cov_port = np.linalg.pinv(self.p_phi_xi_cov_inv + d_xi @ d_xi.T)
        # new p_phi_xi_df
        p_xi_df_post = self.p_phi_xi_df + n_samples

        m_psi_xi = invwishart.rvs(p_xi_df_post, p_xi_cov_port)

        # TODO: Set Zeros where it needed?
        # Set parameters values
        return None

    def gibbs_psi_y(self):
        """
        Gibbs sampling of Phi_y"""
        pass

    def calc_z_bounds(self, params=None):
        """
        This function calculates alpha-values for boundaries
        :return:
        """
        if params is None:
            params = self.param_val
        return {0: [-10, 0, 10], 1: [-10, 0, 10]}

    def gibbs_z(self, params=None):
        """
        Gibbs sampling of Z variables
        :param params:
        :return:
        """
        if params is None:
            params = self.param_val
        pass

    def gibbs_y(self, params=None):
        """
        Gibbs sampling of Y variables
        :param params:
        :return:
        """
        if params is None:
            params = self.param_val

        """ Sampling Y -- latent variables of genotypes
        :return Y sample as matrix """
        d_y = np.zeros((self.n_g, self.n_obs))
        for i in range(self.n_obs):
            y_tmp = \
                st.multivariate_normal(mean=np.zeros(self.n_g),
                                       cov=self.get_matrix(SEMmx.PHI_Y, params))
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
        m_lambda = self.get_matrix(params, SEMmx.LAMBDA)
        m_kappa = self.get_matrix(params, SEMmx.KAPPA)
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


    def gibbs_theta_delta(self, params):
        pass

    def gibbs_spart(self, params):
        pass

    def gibbs_theta_eps(self, params):
        pass

    def gibbs_mpart(self, params):
        pass
    # =========================================================================
    # OLD Masha's version
    # =========================================================================



    # def samp_omega(self,params):
    #     """
    #     Sampling Omega -- latent variables
    #     result: new sample Omega"""
    #     m_inv_sigma_x = np.linalg.pinv(SEMModelFull.get_sigma_x(params))
    #     m_ins_sigma_omega = np.linalg.pinv(SEMModelFull.get_sigma_omega(params))
    #     m_lambda = SEMModelFull.get_matrix(params,"Lambda")
    #     m_kappa = SEMModelFull.get_matrix(params, "Kappa")
    #     m_inv_q=m_lambda.T @ m_inv_sigma_x @ m_lambda + m_ins_sigma_omega
    #     m_q=np.linalg.pinv(m_inv_q)
    #     # X  матрица, столбцы X - элементы веборки
    #     sample_size=self.m_X.shape[1]
    #     for i in range(sample_size):
    #         v_x=self.m_x[:,i]
    #         v_y=self.m_y[:,i]
    #         v_q=m_lambda.T @ m_inv_sigma_x @ (v_x - m_kappa @ v_y)
    #         m_omega[:,i]= np.random.normal(m_q @ v_q, m_q)
    #     return m_omega

    def samp_y(self,params):
        """ Sampling Y -- latent variables of genotypes
        :return Y sample as matrix """
        samp_size=self.m_y.shape[0]
        v_size=self.m_y.shape[1]
        for i in range(samp_size):
            m_pre_y[:, i] = np.random.normal(np.zeros(rows), SEMModelFull.get_matrix(params,"Psi_y"))
            for j in range(v_size):
                 if(self.m_g[j,i]==1 and m_pre_y[j,i]>0) or (self.m_g[j,i]==0 and m_pre_y[j,i]<0):
                     m_y[j, i] = m_pre_y[j, i]
                 else:
                     m_y[j, i] = 0
        return m_y

    def samp_psi_xi(self, params):
        """Sampling covariance matrix Psi_xi
         :return matrix Psi_xi,
                 parameters of Wishard distibution of Psi_xi:
                 p_psi_xi_df - degrees of freedom,
                 p_psi_xi_cov - matrix """
        m_omega_xi = self.get_omega_xi(self, params)
        sample_size = m_omega_xi.shape[1]
        p_psi_xi_cov = np.linalg.pinv(np.linalg.pinv(self.p_psi_xi_cov) + m_omega_xi @ m_omega_xi.T)  # new p_psi_xi_cov
        p_psi_xi_df = self.p_psi_xi_df + sample_size  # new p_psi_df
        m_psi_xi = invwishart.logpdf(SEMModelFull.get_matrix(params, 'Psi_xi'), p_psi_xi_df, p_psi_xi_cov)
        return m_psi_xi, p_psi_xi_df, p_psi_xi_cov

    def samp_psi_y(self, params):
        """Sampling covariance matrix Psi_y
         :return matrix Psi_y,
                parameters of Wishard distibution of Psi_y:
                p_psi_y_df - degrees of freedom,
                 p_psi_y_cov - matrix """

        sample_size = self.m_y.shape[1]
        p_psi_y_cov = np.linalg.pinv(np.linalg.pinv(self.p_psi_y_cov) + self.m_y @ self.m_y.T)
        p_psi_y_df = self.p_psi_y_df + sample_size  # new p_psi_df
        m_psi_y = invwishart.logpdf(SEMModelFull.get_matrix(params, 'Psi_y'), p_psi_y_df, p_psi_y_cov)
        return m_psi_y, p_psi_y_df, p_psi_y_cov

    def samp_theta_delta_b_pi_gamma(self, params):
        """
        Sampling covariance matrixes Theta_delta, B, Pi, Gamma
         :return matrix Theta_delta,
                 parameter of Gamma distribution of Theta_delta: p_theta_delta_alpha,
                 matrixes B, Pi, Gamma,
                 parameters of Normal distribution of matrix(B, Pi, Gamma): p_b_pi_gamma_means, p_b_pi_gamma_covs """

        sample_size = self.m_x.shape[1]
        p_theta_delta_alpha = self.p_theta_delta_alpha + sample_size / 2
        p_theta_delta_beta = self.p_theta_delta_beta
        m_theta_delta = invgamma.logpdf(SEMModelFull.get_matrix('Theta_delta'), p_theta_delta_alpha, p_theta_delta_beta)
        m_omega_y = self.get_omega_y(self, params)
        m_omega_eta = self.get_omega_eta(self, params)
        p_b_pi_gamma_means = zeros((self.p_b_pi_gamma_means.shape[0],
                                    self.p_b_pi_gamma_means.shape[1],
                                    self.p_b_pi_gamma_means.shape[2]))
        p_b_pi_gamma_covs = zeros((self.p_b_pi_gamma_covs.shape[0],
                                   self.p_b_pi_gamma_covs.shape[1],
                                   self.p_b_pi_gamma_covs.shape[2]))
        rows = self.p_b_pi_gamma_means.shape[0]
        b_pi_gamma = SEMModelFull.get_matrix('B_Pi_Gamma')
        for j in range(rows):
            p_b_pi_gamma_covs[j] = np.linalg.pinv(m_omega_y @ m_omega_y.T + np.linalg.pinv(self.p_b_pi_gamma_covs[j]))
            p_b_pi_gamma_means[j] = p_b_pi_gamma_covs[j] @ \
                                    (m_omega_y @ m_omega_eta[j].T + \
                                     np.linalg.pinv(self.p_b_pi_gamma_covs @ self.p_b_pi_gamma_means))
            b_pi_gamma[j] = multivariate_normal.logpdf(b_pi_gamma[j], p_b_pi_gamma_means[j],
                                                       m_theta_delta[j, j] * p_b_pi_gamma_covs[j])
        #b_pi_gamma_cols=b_pi_gamma.shape[1]
        #m_b_old=SEMModelFull.get_matrix(params,"B")
        #b_cols=m_b_old.shape[1]
        #p_cols = [j for k, j in b_pi_gamma_cols if k in b_cols]
        m_b=self.get_b_from_b_pi_gamma(self,params)
        m_pi = self.get_pi_from_b_pi_gamma(self, params)
        m_gamma = self.get_gamma_from_b_pi_gamma(self, params)
        return m_theta_delta, p_theta_delta_alpha, m_b, m_pi, m_gamma , p_b_pi_gamma_means, p_b_pi_gamma_covs

    def sample_lambda_kappa_epsilon(self, params):
        """
        Sampling covariance matrixes Theta_eps, Lambda, Kappa and parameters of distribution
        :return matrix Theta_eps,
                parameter of Gamma distribution of Theta_eps: p_theta_eps_alpha,
                matrixes Lambda, Kappa,
                parameters of Normal distribution of matrix(Lambada, Kappa): p_lambda_kappa_means, p_lambda_kappa_covs """

        sample_size = self.m_x.shape[1]
        p_theta_eps_alpha = self.p_theta_eps_alpha + sample_size / 2
        p_theta_eps_beta = self.p_theta_eps_beta
        m_theta_eps = invgamma.logpdf(SEMModelFull.get_matrix('Theta_eps'), p_theta_eps_alpha, p_theta_eps_beta)
        m_x = self.m_x
        m_x_y = self.get_x_y(self, params)
        p_lambda_kappa_means = zeros((self.p_lambda_kappa_means.shape[0],
                                      self.p_lambda_kappa_means.shape[1],
                                      self.p_lambda_kappa_means.shape[2]))
        p_lambda_kappa_covs = zeros((self.p_lambda_kappa_covs.shape[0],
                                     self.p_lambda_kappa_covs.shape[1],
                                     self.p_lambda_kappa_covs.shape[2]))

        rows = self.p_b_pi_gamma_means.shape[0]
        m_lambda_kappa = SEMModelFull.get_matrix('B_Pi_Gamma')
        for j in range(rows):
            p_lambda_kappa_covs[j] = np.linalg.pinv(m_x_y @ m_x_y.T + np.linalg.pinv(self.p_lambda_kappa_covs[j]))
            p_lambda_kappa_means[j] = p_lambda_kappa_covs[j] @ \
                                      (m_x_y @ m_x[j].T + \
                                       np.linalg.pinv(self.p_lambda_kappa_covs @ self.p_lambda_kappa_means))
            m_lambda_kappa[j] = multivariate_normal.logpdf(m_lambda_kappa[j], p_lambda_kappa_means[j],
                                                           m_theta_eps[j, j] * p_lambda_kappa_covs[j])
        m_lambda=self.get_lambda_from_lambda_kappa(self,params)
        m_kappa = self.get_kappa_from_lambda_kappa(self, params)
        return m_theta_eps, m_lambda, m_kappa, p_lambda_kappa_means, p_lambda_kappa_covs

    def get_lambda_from_lambda_kappa(self,params):
        pass
    def get_kappa_from_lambda_kappa(self,params):
        pass
    def get_x_y(self, params):
        pass

    def samp_bounds(self, params):
        """ Sampling new bounds for vector z
        :return new bounds as array"""
        prob_bounds = self.prob_bounds
        m_theta_eps_v = SEMModelFull.get_theta_eps_v(params)
        for j in range(prob_bounds.shape[0]):
            sum_prob_bounds = 0
            for r in range(prob_bounds[j].shape[0]):
                sum_prob_bounds += prob_bounds[j, r]
                bounds[j, r] = erfinv(sum_prob_bounds) * sqrt(2 * m_theta_eps_v[j, j])

        return bounds

    def samp_z(self, params):
        """ Sampling Z
         :return Z sample as matrix"""
        for i in range(self.m_z.shape[0]):
            m_pre_z[:, i] = np.random.normal(np.zeros(self.m_z.shape[1]), SEMModelFull.get_sigma_z(params))
            for j in range(self.m_z.shape[1]):
                categor = self.m_v[j, i]
                if (m_pre_z[j, i] > self.bounds[categor - 1, j]) and (m_pre_z[j, i] <= self.bounds[categor, j]):
                    m_z[j, i] = m_pre_y[j, i]
                else:
                    m_z[j, i] = 0
        return m_z

    def prior_m_x(self):
        pass

    def prior_m_y(self):
        pass

    def prior_m_g(self):
        pass

    def prior_m_z(self):
        pass

    def prior_m_v(self):
        pass

    def prior_m_omega(self):
        pass

    def prior_params_psi_xi(self, params):
        """

        :param params:
        :return:
        """
        #return p_psi_y_df, p_psi_y_cov
        pass

    def prior_z_bounds(self,params):
        pass

    def prior_prob_bounds(self):
        pass

    def prior_p_lambda_kappa(self, params):
        #return p_lambda_kappa_means, p_lambda_kappa_covs
        pass

    def prior_p_theta_eps(self):
        pass

    def get_omega_xi(self,params):
        pass

    def get_omega_y(self, params):
        pass

    def get_omega_eta(self, params):
        pass

    def get_b_from_b_pi_gamma(self, params):
        pass

    def get_pi_from_b_pi_gamma(self, params):
        pass

    def get_gamma_from_b_pi_gamma(self,params):
        pass