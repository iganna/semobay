from sem_model_full import SEMModelFull
from sem_model import SEMData
from scipy.stats import invwishart, invgamma, wishart, norm, uniform, multivariate_normal
from scipy.special import erfinv


class SEMOptBayesFull():
    def __init__(self, mod: SEMModelFull, data: ):

        # Set attributes for parameters
        sels.param_val

        self.n_sample_omega_pos = {}
        self.n_sample_omega_val = []
        # Prior samples
        self.m_x = self.prior_m_x()
        self.m_g = self.prior_m_g()
        self.m_y = self.prior_m_y()
        self.m_omega = self.prior_m_omega()
        self.m_z=self.prior_m_z(self)
        self.m_v = self.prior_m_v(self)
        # Prior bounds
        self.bounds=self.prior_bounds(self,params)
        self.prob_bounds=self.prior_prob_bounds(self)
        # Parameters for prior distributions
        self.p_psi_xi_df, self.p_psi_xi_cov = self.prior_params_psi_xi(self,params)
        self.p_theta_delta_alpha, self.p_theta_delta_beta = self.prior_p_theta_delta(params)
        self.p_b_pi_gamma_means, self.p_b_pi_gamma_covs=self.prior_p_b_pi_gamma(params)
        self.p_lambda_kappa_means, self.p_lambda_kappa_covs = self.prior_p_lambda_kappa(self,params)



    def optimise(self):
        params_init = np.array(self.params)
        params = np.array(params_init)

        for _ in range(1000):



            params =self.samp_psi_xi(params_init)

            params[2]=self.samp_phi_y(self, params)

            params[3], self.p_b_pi_gamma_means, params[4], params[5], params[6], self.p_b_pi_gamma_covs = \
                self.samp_theta_delta_b_pi_gamma(params)
            params[7], self.p_lambda_kappa_means,  params[8], params[9],self.p_lambda_kappa_covs=\
                self.samp_lambda_k_epsilon(params)

            self.m_omega = self.samp_omega(params)
            self.m_y = self.samp_y(params)
            self.bounds=self.samp_bounds(params)
            self.m_z=self.samp_z(params)

    def samp_omega(self,params):
        """
        Sampling Omega -- latent variables
        result: new sample Omega"""
        m_inv_sigma_x=np.linalg.pinv(SEMModelFull.get_sigma_x(params))
        m_ins_sigma_omega=np.linalg.pinv(SEMModelFull.get_sigma_omega(params))
        m_lambda = SEMModelFull.get_matrix(params,"Lambda")
        m_kappa = SEMModelFull.get_matrix(params, "Kappa")
        m_inv_q=m_lambda.T @ m_inv_sigma_x @ m_lambda + m_ins_sigma_omega
        m_q=np.linalg.pinv(m_inv_q)
        # X  матрица, столбцы X - элементы веборки
        sample_size=self.m_X.shape[1]
        for i in range(sample_size):
            v_x=self.m_x[:,i]
            v_y=self.m_y[:,i]
            v_q=m_lambda.T @ m_inv_sigma_x @ (v_x - m_kappa @ v_y)
            m_omega[:,i]= normal.rvs(m_q @ v_q, m_q)
        return m_omega

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
        # b_pi_gamma_cols=b_pi_gamma.shape[1]
        # m_b_old=SEMModelFull.get_matrix(params,"B")
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
        #return p_psi_y_df, p_psi_y_cov
        pass
    def prior_bounds(self,params):
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