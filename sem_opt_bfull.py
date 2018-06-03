import numpy as np
from sem_model_full import SEMModelFull, SEMmx
from sem_model import SEMData
import itertools as it
from scipy.special import erfinv
import scipy.stats as st
import collections



class SEMOptBayesFull():
    def __init__(self, mod, data, param_prior):

        # -------------------------------------------
        # Get data profiles "d_" means "data"
        # -------------------------------------------
        self.d_g = mod.d_g
        self.d_u = mod.d_u
        self.d_v = mod.d_v

        # -------------------------------------------
        self.n_obs = self.d_g.shape[0]  # Number of observatioons (samples)
        self.n_eta = len(mod.d_vars['eta'])
        self.n_xi = len(mod.d_vars['xi'])
        self.n_omega = self.n_eta + self.n_xi
        self.n_g = len(mod.d_vars['g'])
        self.n_x = len(mod.d_vars['x'])
        self.n_z = len(mod.d_vars['v'])
        self.n_spart = self.n_eta + self.n_xi + self.n_g
        self.n_mpart = self.n_eta + self.n_xi + self.n_g

        # -------------------------------------------
        self.get_matrix = mod.get_matrix
        self.d_xi = np.zeros((self.n_obs, self.n_xi))
        self.d_eta = np.zeros((self.n_obs, self.n_eta))
        self.d_y = np.zeros((self.n_obs, self.n_g))


        # -------------------------------------------
        # Get parameters: initial values and annotation
        # -------------------------------------------
        # Set attributes for parameters
        self.param_val = np.array(mod.param_val)
        # It is a dictionary, that explains parameters:
        self.param_pos = mod.param_pos
        self.param_fix = mod.param_fix
        self.param_prior = None
        self.load_prior_params(param_prior)

        self.mcmc = [np.array(self.param_val)]


        # -------------------------------------------
        self.coefs_spart = self.get_coefs_spart()
        self.coefs_mpart = self.get_coefs_mpart()


        # Proportion of samples of the same level
        # Cummulative fractions to calculate CDFs
        self.z_cumm_fract = self.get_z_cumm_fract()
        self.z_counts = self.get_z_counts()


        # -------------------------------------------
        # Parameters for prior distributions
        # -------------------------------------------
        # Phi matrices ~ Inverse Wishart:
        self.p_phi_xi_df, self.p_phi_xi_cov_inv = self.get_params_phi_xi()
        self.p_phi_y_df, self.p_phi_y_cov_inv = self.get_params_phi_y()



        # Theta matrices ~ Inverse Gamma
        self.p_theta_delta_alpha, self.p_theta_delta_beta = \
            self.get_params_theta_delta()
        self.p_theta_eps_alpha, self.p_theta_eps_beta = \
            self.get_params_theta_eps()

        # Parameters for normal distribution of path coefficients in
        # Structural part

        self.p_spart_mean, self.p_spart_cov_inv = self.get_params_spart()
        self.p_spart_loc = [cov_inv @ mean
                            for mean, cov_inv in zip(self.p_spart_mean,
                                                     self.p_spart_cov_inv)]
        self.p_spart_qform = [mean.T @ loc
                              for mean, loc in zip(self.p_spart_mean,
                                                   self.p_spart_loc)]

        # Measurement part
        self.p_mpart_mean, self.p_mpart_cov_inv = self.get_params_mpart()
        self.p_mpart_loc = [cov_inv @ mean
                            for mean, cov_inv in zip(self.p_mpart_mean,
                                                     self.p_mpart_cov_inv)]
        self.p_mpart_qform = [mean.T @ loc
                              for mean, loc in zip(self.p_mpart_mean,
                                                   self.p_mpart_loc)]

        # -----------------------------------------------
        # order of Binary and Ordinal variables
        # -----------------------------------------------
        self.idx_z = None
        self.idx_y = None

    @property
    def d_omega(self):
        return np.concatenate((self.d_eta, self.d_xi), axis=1)

    @property
    def d_spart(self):
        return np.concatenate((self.d_eta, self.d_xi, self.d_y), axis=1)

    @property
    def d_mpart(self):
        return np.concatenate((self.d_eta, self.d_xi, self.d_y), axis=1)

    @property
    def d_x(self):
        return np.concatenate((self.d_u, self.d_z), axis=1)

    def optimise(self):

        params = np.array(self.param_val)

        n_iter = 20000
        print(n_iter)
        for _ in range(n_iter):

            # --------------------------------------------------------
            # Initial values for z-boundaries
            # --------------------------------------------------------
            self.z_bounds = self.calc_z_bounds()

            # --------------------------------------------------------
            # Sample values for latent variables
            # --------------------------------------------------------
            # self.d_z = self.gibbs_z()
            # self.d_y = self.gibbs_y()

            self.d_z = self.gibbs_z_new()
            self.d_y = self.gibbs_y_new()
            self.d_eta, self.d_xi = self.gibbs_omega()

            # --------------------------------------------------------
            # Sample covariance matrices Phi_xi and Phi_y
            # --------------------------------------------------------
            self.gibbs_phi_xi()
            self.gibbs_phi_y()

            # --------------------------------------------------------
            # Samples all Parameters (errors first)
            # --------------------------------------------------------
            # Structural Part
            self.gibbs_spart()
            # Measurement Part
            self.gibbs_mpart()
            # Fix parameter values
            self.fix_param_values()

            # --------------------------------------------------------
            # Remember values of parameters after each iteration
            # --------------------------------------------------------

            self.mcmc = np.append(self.mcmc, [self.param_val], axis=0)
            print(self.mcmc.shape)

        return self.mcmc

    def load_prior_params(self, param_prior):
        self.param_prior = param_prior

    def get_params_phi_xi(self):
        """
        Get prior parameters for Exogenous latent variables:
        df and scale
        :return:
        """
        if self.param_prior is None:
            return 5, np.identity(self.n_xi)

        m_phi = self.get_matrix(SEMmx.PHI_XI, self.param_prior)
        r = len([param_id for mx_type, _, _, param_id in self.param_pos
                 if mx_type in {SEMmx.LAMBDA_V_XI,
                                SEMmx.LAMBDA_U_XI,
                                SEMmx.GAMMA}])
        pho = r + 4

        m_r_inv = m_phi * (pho - self.n_xi - 1)
        return pho, m_r_inv


    def get_params_phi_y(self):
        """
        Get prior parameters for latent variables correspond to genetic:
        df and scale
        :return:
        """
        if self.param_prior is None:
            return 5, np.identity(self.n_g)

        m_phi = self.get_matrix(SEMmx.PHI_Y, self.param_prior)
        r = len([param_id for mx_type, _, _, param_id in self.param_pos
                 if mx_type in {SEMmx.PI, SEMmx.KAPPA}])
        pho = r + 4

        m_r_inv = m_phi * (pho - self.n_g - 1)
        return pho, m_r_inv

    def get_params_theta_delta(self):
        """
        Get patameters for InvGamma distribution
        :return: alpha, beta
        """
        if self.param_prior is None:
            return np.ones(self.n_eta) * 9, np.ones(self.n_eta) * 4

        m_theta = np.diag(self.get_matrix(SEMmx.THETA_DELTA, self.param_prior))
        alpha = np.ones(self.n_eta) * 3
        beta = (alpha - 1) * m_theta
        return alpha, beta


    def get_params_theta_eps(self):
        """
        Get patameters for InvGamma distribution
        :return:
        """
        if self.param_prior is None:
            return np.ones(self.n_x) * 9, np.ones(self.n_x) * 4

        m_theta = np.diag(self.get_matrix(SEMmx.THETA_EPS, self.param_prior))
        alpha = np.ones(self.n_x) * 3
        beta = (alpha - 1) * m_theta
        return alpha, beta


    def get_params_spart(self):
        """
        Get parameters of Normal Distribution for path coefficients in
        the Structural part
        :return: mean value and INVERSE covariance matrix
        """
        res_mean = []
        res_invcov = []
        for i in range(self.n_eta):
            n_terms = len(self.coefs_spart[i])

            # For informative
            if self.param_prior is None:
                res_mean += [np.ones(n_terms) * 0.8]
            else:
                res_mean += [np.array([self.param_prior[param_id]
                             for _, param_id in self.coefs_spart[i]])]

            res_invcov += [np.identity(n_terms)]


            # # For non-informative
            # res += (np.ones(n_terms) * 0.8,
            #         np.zeros(n_terms))

            # self. = np.linalg.inv(self.p_spart_cov)

        return res_mean, res_invcov

    def get_params_mpart(self):
        """
        Get parameters of Normal Distribution for factor loadings
        in the Measurement part
        :return: mean value and INVERSE covariance matrix
        """
        res_mean = []
        res_invcov = []
        for i in range(self.n_x):
            n_terms = len(self.coefs_mpart[i])

            # For informative
            if self.param_prior is None:
                res_mean += [np.ones(n_terms) * 0.8]
            else:
                res_mean += [np.array([self.param_prior[param_id]
                             for _, param_id in self.coefs_mpart[i]])]

            res_invcov += [np.identity(n_terms)]

            # # For non-informative
            # res += (np.ones(n_terms) * 0.8,
            #         np.zeros(n_terms))
        return res_mean, res_invcov

    def get_z_cumm_fract(self):
        """
        Returm cummulative fractions
        :return:
        # """
        z_cumm_fract = []
        for i, v in enumerate(self.d_v.T):
            v_sort = np.array(v)
            v_sort.sort()
            unique, counts = np.unique(v_sort, return_counts=True)
            counts = counts / sum(counts)
            cumm_counts = np.cumsum(counts)

            z_cumm_fract += [cumm_counts]

        # here is an example
        return z_cumm_fract

    def get_z_counts(self):
        """
        Returm cummulative fractions
        :return:
        # """
        z_counts = []
        for i, v in enumerate(self.d_v.T):
            v_sort = np.array(v)
            v_sort.sort()
            unique, counts = np.unique(v_sort, return_counts=True)
            z_counts += [counts]

        # here is an example
        return z_counts

    def get_y_counts(self):
        """
        Returm cummulative fractions
        :return:
        # """
        y_counts = []
        for i, v in enumerate(self.d_g.T):
            v_sort = np.array(v)
            v_sort.sort()
            unique, counts = np.unique(v_sort, return_counts=True)
            y_counts += [counts]

        # here is an example
        return y_counts

    def get_coefs_spart(self):
        """

        :return:
        """
        coefs = []
        for irow in range(self.n_eta):
            coefs_row = []
            for mx_type, pos1, pos2, param_id in self.param_pos:
                if mx_type is not SEMmx.SPART:
                    continue
                if pos1 != irow:
                    continue
                coefs_row += [(pos2, param_id)]
            coefs += [coefs_row]

        return coefs

    def get_coefs_mpart(self):
        """

        :return:
        """
        coefs = []
        for irow in range(self.n_x):
            coefs_row = []
            for mx_type, pos1, pos2, param_id in self.param_pos:
                if mx_type is not SEMmx.MPART:
                    continue
                if pos1 != irow:
                    continue
                coefs_row += [(pos2, param_id)]
            coefs += [coefs_row]

        return coefs
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

        if self.n_xi == 0:
            return
        # Posterior parameters
        # new p_phi_xi_df
        p_xi_cov_post = self.p_phi_xi_cov_inv + d_xi.T @ d_xi
        # new p_phi_xi_df
        p_xi_df_post = self.p_phi_xi_df + n_samples

        m_phi_xi = st.invwishart.rvs(scale=p_xi_cov_post, df=p_xi_df_post)

        if not isinstance(m_phi_xi, collections.Iterable):
            m_phi_xi = [[m_phi_xi]]

        for mx_type, pos1, pos2, param_id in self.param_pos:
            if mx_type is SEMmx.PHI_XI:
                self.param_val[param_id] = m_phi_xi[pos1][pos2]

    def gibbs_phi_y(self):
        """
        Gibbs sampling of Phi_y
           Sampling covariance matrix Phi_y
         :return matrix Phi_y,
                parameters of Wishard distibution of Phi_y:
                p_phi_y_df - degrees of freedom,
                 p_phi_y_cov - matrix """

        if self.n_g == 0:
            return
        d_y = self.d_y  # This is already sampled by the Gibbs sampling
        n_samples = self.n_obs

        # Posterior parameters
        # new p_phi_y_df
        p_y_cov_post = self.p_phi_y_cov_inv + d_y.T @ d_y
        # new p_phi_y_df
        p_y_df_post = self.p_phi_y_df + n_samples

        m_phi_y = st.invwishart.rvs(scale=p_y_cov_post, df=p_y_df_post)

        if not isinstance(m_phi_y, collections.Iterable):
            m_phi_y = [[m_phi_y]]

        for mx_type, pos1, pos2, param_id in self.param_pos:
            if mx_type is SEMmx.PHI_Y:
                self.param_val[param_id] = m_phi_y[pos1][pos2]

    def calc_z_bounds(self):
        """
        This function calculates alpha-values for boundaries
        :return:
        """
        if self.n_z == 0:
            return []
        m_sigma_z = self.get_matrix(SEMmx.SIGMA_Z, self.param_val)
        percentiles = []
        for fractions, variance in zip(self.z_cumm_fract, np.diag(m_sigma_z)):
            percentiles += [st.norm.ppf(fractions,
                                       scale=variance**(1/2))]

        return percentiles

    def gibbs_z(self):
        """
        Gibbs sampling of Z variables
        :param params:
        :return:
        """
        def get_ord_value(value, bounds):
            for category, bound in enumerate(reversed(bounds)):
                if value < bound:
                    return category

        d_z = np.zeros((self.n_obs, self.n_z))
        if self.n_z == 0:
            return d_z

        m_sigma_z = self.get_matrix(SEMmx.SIGMA_Z, self.param_val)

        for i in range(self.n_obs):
            z_tmp = \
                st.multivariate_normal.rvs(mean=np.zeros(self.n_z),
                                           cov=m_sigma_z)
            d_z[i, :] = z_tmp

        # Chack for correct class
        for i, j in it.product(range(self.n_obs), range(self.n_z)):

            value_norm = d_z[i, j]
            value_smpl_ord = get_ord_value(value_norm, self.z_bounds[j])
            value_ord = self.d_v[i, j]

            if value_smpl_ord != value_ord:
                d_z[i, j] = 0

        return d_z

    # def gibbs_z_new(self):
    #     """
    #     Gibbs sampling of Z variables
    #     :param params:
    #     :return:
    #     """
    #     def get_ord_value(value, bounds):
    #         for category, bound in enumerate(reversed(bounds)):
    #             if value < bound:
    #                 return category
    #
    #     d_z = np.zeros((self.n_obs, self.n_z))
    #     if self.n_z == 0:
    #         return d_z
    #
    #     m_sigma_z = self.get_matrix(SEMmx.SIGMA_Z, self.param_val)
    #     m_lambda_v = \
    #         np.concatenate((self.get_matrix(SEMmx.LAMBDA_V_ETA, self.param_val),
    #                         self.get_matrix(SEMmx.LAMBDA_V_XI, self.param_val)),
    #                        axis=1)
    #     m_kappa_v = self.get_matrix(SEMmx.KAPPA_V, self.param_val)
    #     d_omega = self.d_omega
    #     d_y = self.d_y
    #
    #     for i, j in it.product(range(self.n_obs), range(self.n_z)):
    #
    #
    #         z_mean = m_lambda_v[j, :] @ d_omega[i, :].T + \
    #                  m_kappa_v[j, :] @ d_y[i, :].T
    #         z_cov = m_sigma_z[j][j]
    #
    #
    #         z_tmp = \
    #             st.multivariate_normal.rvs(mean=z_mean,
    #                                        cov=z_cov,
    #                                        size=1)
    #         # z_tmp.sort()
    #         #
    #         # # define the order
    #         # tmp_sample = self.d_v[:, i] + np.random.rand(self.n_obs)/100
    #         # tmp_dict = {x: ind for ind, x in enumerate(sorted(tmp_sample))}
    #         # idx = [tmp_dict[val] for val in tmp_sample]
    #         # d_z[:, i] = [z_tmp[j] for j in idx]
    #
    #         d_z[i, j] = z_tmp
    #
    #
    #     # Chack for correct class
    #     for i, j in it.product(range(self.n_obs), range(self.n_z)):
    #
    #         value_norm = d_z[i, j]
    #         value_smpl_ord = get_ord_value(value_norm, self.z_bounds[j])
    #         value_ord = self.d_v[i, j]
    #
    #         if value_smpl_ord != value_ord:
    #             d_z[i, j] = 0
    #
    #     return d_z


    def gibbs_z_new(self):
        """
        Gibbs sampling of Z variables
        :param params:
        :return:
        """

        d_z = np.zeros((self.n_obs, self.n_z))
        if self.n_z == 0:
            return d_z

        m_sigma_z = self.get_matrix(SEMmx.SIGMA_Z, self.param_val)

        for i in range(self.n_z):
            z_tmp = \
                st.multivariate_normal.rvs(mean=np.zeros(1),
                                           cov=m_sigma_z[i][i],
                                           size=self.n_obs)
            z_tmp.sort()

            # define the order
            # if self.idx_z is None:
            tmp_sample = self.d_v[:, i] + np.random.rand(self.n_obs)/100
            tmp_dict = {x: ind for ind, x in enumerate(sorted(tmp_sample))}
            self.idx_z = [tmp_dict[val] for val in tmp_sample]

            d_z[:, i] = [z_tmp[j] for j in self.idx_z]

        return d_z

    def gibbs_y_new(self):

        d_y = np.zeros((self.n_obs, self.n_g))
        if self.n_g == 0:
            return d_y

        m_sigma_y = self.get_matrix(SEMmx.PHI_Y, self.param_val)

        for i in range(self.n_g):
            y_tmp = \
                st.multivariate_normal.rvs(mean=np.zeros(1),
                                           cov=m_sigma_y[i][i],
                                           size=self.n_obs)
            y_tmp.sort()

            # define the order
            # if self.idx_y is None:
            tmp_sample = self.d_g[:, i] + np.random.rand(self.n_obs) / 100
            tmp_dict = {x: ind for ind, x in enumerate(sorted(tmp_sample))}
            self.idx_y = [tmp_dict[val] for val in tmp_sample]

            d_y[:, i] = [y_tmp[j] for j in self.idx_y]

        return d_y

    def gibbs_y(self):
        """
        Gibbs sampling of Y variables
        Sampling Y -- latent variables of genotypes
        :return Y sample as matrix """

        d_y = np.zeros((self.n_obs, self.n_g))
        if self.n_g == 0:
            return d_y

        m_phi_y = self.get_matrix(SEMmx.PHI_Y, self.param_val)
        for i in range(self.n_obs):
            y_tmp = \
                st.multivariate_normal.rvs(mean=np.zeros(self.n_g),
                                           cov=m_phi_y)

            if not isinstance(y_tmp, collections.Iterable):
                y_tmp = [y_tmp]

            y_new = [y if (g == 1) == (y > 0) else 0
                     for y, g in zip(y_tmp, self.d_g[i, :])]
            d_y[i, :] = y_new
        return d_y

    def gibbs_omega(self):
        """
        Sampling Omega -- latent variables
        result: new sample Omega"""
        d_omega = np.zeros((self.n_obs, self.n_omega))
        if self.n_omega == 0:
            d_eta = d_omega[:, 0:self.n_eta]
            d_xi = d_omega[:, self.n_eta:]
            return d_eta, d_xi


        # # MASHA
        # m_tmp = self.get_matrix(SEMmx.THETA_EPS, self.param_val) + \
        #         self.get_matrix(SEMmx.KAPPA, self.param_val) @ \
        #         self.get_matrix(SEMmx.PHI_Y, self.param_val) @ \
        #         self.get_matrix(SEMmx.KAPPA, self.param_val)
        # m_inv_sigma_x = np.linalg.pinv(m_tmp)


        # GOOD
        m_inv_sigma_x = np.linalg.pinv(self.get_matrix(SEMmx.THETA_EPS,
                                                       self.param_val))

        m_inv_sigma_omega = np.linalg.pinv(self.get_matrix(SEMmx.SIGMA_OMEGA,
                                                           self.param_val))



        m_lambda = self.get_matrix(SEMmx.LAMBDA, self.param_val)
        m_kappa = self.get_matrix(SEMmx.KAPPA, self.param_val)
        m_inv_q = m_lambda.T @ m_inv_sigma_x @ m_lambda + m_inv_sigma_omega
        m_q = np.linalg.pinv(m_inv_q)

        for i in range(self.n_obs):
            x = self.d_x[i, :]  # Do not need to transpose
            y = self.d_y[i, :]  # Do not need to transpose
            q = m_lambda.T @ m_inv_sigma_x @ (x - m_kappa @ y)
            d_omega[i, :] = st.multivariate_normal.rvs(mean=m_q @ q,
                                                       cov=m_q)

        d_eta = d_omega[:, 0:self.n_eta]
        d_xi = d_omega[:, self.n_eta:]
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

        if self.n_eta == 0:
            return

        # Sampling Theta and (Beta, Gamma, Pi) by rows
        for irow in range(self.n_eta):
            pos_of_coef = [pos2 for pos2, param_id in self.coefs_spart[irow]]
            id_of_params = [param_id for pos2, param_id in
                            self.coefs_spart[irow]]
            d_tmp = d_spart[:, pos_of_coef]
            # Calculate auxiliary variables
            a_cov_inv = self.p_spart_cov_inv[irow] + d_tmp.T @ d_tmp
            a_cov = np.linalg.inv(a_cov_inv)
            a_mean = a_cov @ (self.p_spart_loc[irow] +
                              d_tmp.T @ d_eta[:, irow].T)

            # Calculate new parameters of InvGamma dna InvWishart
            p_alpha = self.p_theta_delta_alpha[irow] + n_obs / 2
            p_beta = self.p_theta_delta_beta[irow] + 1 / 2 * \
                     (d_eta[:, irow].T @ d_eta[:, irow] -
                      a_mean.T @ a_cov_inv @ a_mean +
                      self.p_spart_qform[irow])

            value_of_theta = st.invgamma.rvs(a=p_alpha,
                                             scale=p_beta)
            value_of_coef = \
                st.multivariate_normal.rvs(mean=a_mean,
                                           cov=a_cov*value_of_theta)

            if not isinstance(value_of_coef, collections.Iterable):
                value_of_coef = [value_of_coef]

            # -------------------------------
            # Set new parameters values
            # -------------------------------

            for mx_type, pos1, pos2, param_id in self.param_pos:
                if mx_type is SEMmx.THETA_DELTA and pos1 == irow:
                    self.param_val[param_id] = value_of_theta

            for param_id, value in zip(id_of_params, value_of_coef):
                self.param_val[param_id] = value

    def gibbs_mpart(self):
        """

        :return:
        """

        n_obs = self.n_obs
        d_mpart = self.d_mpart
        d_x = self.d_x
        irow = 0

        # Sampling Theta_eps and (Lambda, Kappa) by rows
        for irow in range(self.n_x):
            pos_of_coef = [pos2 for pos2, param_id in self.coefs_mpart[irow]]
            id_of_params = [param_id for pos2, param_id
                            in self.coefs_mpart[irow]]
            d_tmp = d_mpart[:, pos_of_coef]
            # Calculate auxiliary variables
            a_cov_inv = self.p_mpart_cov_inv[irow] + d_tmp.T @ d_tmp
            a_cov = np.linalg.inv(a_cov_inv)
            a_mean = a_cov @ (self.p_mpart_loc[irow] + d_tmp.T @ d_x[:, irow])

            # Calculate new parameters of InvGamma dna InvWishart
            p_alpha = self.p_theta_eps_alpha[irow] + n_obs / 2
            p_beta = self.p_theta_eps_beta[irow] + 1 / 2 * \
                     (d_x[:, irow].T @ d_x[:, irow] -
                      a_mean.T @ a_cov_inv @ a_mean +
                      self.p_mpart_qform[irow])

            value_of_theta = st.invgamma.rvs(a=p_alpha,
                                            scale=p_beta)
            value_of_coef = \
                st.multivariate_normal.rvs(mean=a_mean,
                                           cov=a_cov*value_of_theta,
                                           size=1)

            if not isinstance(value_of_coef, collections.Iterable):
                value_of_coef = [value_of_coef]

            # -------------------------------
            # Set new parameters values
            # -------------------------------

            for mx_type, pos1, pos2, param_id in self.param_pos:
                if mx_type is SEMmx.THETA_EPS and pos1 == irow:
                    self.param_val[param_id] = value_of_theta
            for param_id, value in zip(id_of_params, value_of_coef):
                self.param_val[param_id] = value

    def fix_param_values(self):
        for param_id, value in self.param_fix:
            self.param_val[param_id] = value