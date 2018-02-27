from sem_model import SEMData, SEMModel
import numpy as np
from functools import partial
from sem_opt_abc import SEMOptABC
from scipy.stats import invwishart, invgamma, wishart, norm, uniform, multivariate_normal
from functools import reduce



class SEMOptBayes(SEMOptABC):

    def __init__(self, mod: SEMModel, data: SEMData, params=None, estimator='EmpBayes'):
        """
        This function....
        :param mod:
        :param data:
        """

        super().__init__(mod, data, estimator, 'none')

        self.loss_func = self.get_loss_function(estimator)

        self.param_pos = mod.param_pos
        self.param_init = self.params

        # Parameters for prior distributions
        self.p_psi_df, self.p_psi_cov = self.prior_params_psi(params)
        self.p_beta_mean, self.p_beta_cov = self.prior_params_beta(params, mod.param_pos)
        self.p_theta_alpha, self.p_theta_beta = self.prior_params_theta(params, mod.param_pos)
        self.p_lambda_mean, self.p_lambda_cov = self.prior_params_lambda(params, mod.param_pos)

        # To save MCMC chain
        self.param_chain = np.array([self.params])


    def loss_functions(self) -> dict:
        """
        Create the dictionary of possible functions
        :return:
        """
        tmp_dict = dict()
        tmp_dict['EmpBayes'] = ((self.log_post_struct, 'Psi'),
                                (self.log_post_struct, 'Beta'),
                                (self.log_post_msrmnt, 'Theta'),
                                (self.log_post_msrmnt, 'Lambda'))

        tmp_dict['Likelihood'] = (self.log_likelihood, ('Psi',
                                                        'Beta',
                                                        'Theta',
                                                        'Lambda'))
        return tmp_dict

    def get_loss_function(self, name):
        loss_dict = self.loss_functions()
        if name in loss_dict.keys():
            return loss_dict[name]
        else:
            raise Exception("SEMOpt_phylo Backend doesn't support loss function {}.".format(name))


    def optimize(self, opt_method=None, bounds=None, alpha=0):
        """ Metropolis hastings algorithm """
        params_init = np.array(self.params)

        params = np.array(params_init)
        for _ in range(1000):
            for log_prob, matrices in self.loss_func:
                params = self.metropolis_hastings(log_prob, matrices, params)
            print(self.log_likelihood(params), self.log_joint(params))
            self.param_chain = np.append(self.param_chain, [params], axis=0)

        self.params = params

        prob_init = self.log_joint(params_init)
        prob_final = self.log_joint(params)
        return prob_init, prob_final


    def metropolis_hastings(self, log_prob, matrices, params):
        params_new = np.array(params)
        for i, pos in self.param_pos.items():
            if pos[0] not in matrices:
                continue
            p = params[i]

            # Try five times to get a parameter which do not make
            # sigma-matrix negatively-defined
            for _ in range(100):
                p_new = norm.rvs(p, 1, 1)
                params_new[i] = p_new
                # If the new parameter value did not make Sigma negative
                # - allow it
                if self.constraint_all(params_new) == 0:
                    break

            # If the required value was not sampled - do not accept it
            if self.constraint_all(params_new) < 0:
                params_new[i] = p
                continue

            # Calculate the Metropolis-Hastings statistics
            mh_log_stat = np.exp(log_prob(params_new) - log_prob(params))
            if mh_log_stat < uniform.rvs(0, 1, 1):
                params_new[i] = p
            else:
                params[i] = params_new[i]

        return params_new


    def log_likelihood(self, params):
        """ Likelihood of data """
        # m_sigma = self.calculate_sigma(params)
        # w = 0
        # for p in self.m_profiles:
        #     w += multivariate_normal.logpdf(p, np.zeros(p.shape), m_sigma)

        m_sigma = self.calculate_sigma(params)
        df = self.m_profiles.shape[0]
        w = wishart.logpdf(self.m_cov, df=df, scale=m_sigma/df)
        return w


    def log_prior_struct(self, params) -> float:
        """ Inverse Whishart distribution of r0 and rho0"""

        ms = self.get_matrices(params)
        inw_psi = invwishart.logpdf(ms['Psi'], self.p_psi_df, self.p_psi_cov)

        """ Normal """
        norm_beta = 0
        for i, p in self.param_pos.items():
            if p[0] != 'Beta':
                continue
            norm_beta += norm.logpdf(params[i],
                                     self.p_beta_mean,
                                     self.p_beta_cov * ms['Psi'][p[1], p[1]])
        return inw_psi + norm_beta


    def log_prior_msrmnt(self, params) -> float:
        """ Inverse Gamma distribution of r0 and rho0"""

        prob_theta = 0
        params_theta = [params[i] for i, p in self.param_pos.items() if p[0] == 'Theta']
        invgamma_specified = partial(invgamma.logpdf,
                                     a=self.p_theta_alpha,
                                     scale=self.p_theta_beta)
        prob_theta = reduce(lambda x, y: x+y,
                            map(invgamma_specified, params_theta))

        """ Normal """
        ms = self.get_matrices(params)
        prob_lambda = 0
        for i, p in self.param_pos.items():
            if p[0] != 'Lambda':
                continue
            prob_lambda += norm.logpdf(params[i],
                                       self.p_lambda_mean,
                                       self.p_lambda_cov * ms['Theta'][p[1], p[1]])
        return prob_theta + prob_lambda



    def log_post_struct(self, params) -> float:
        """ Normal """
        terms = [self.log_prior_struct,
                 self.log_likelihood]
        return reduce(lambda x, y: x + y(params), [0] + terms)


    def log_post_msrmnt(self, params) -> float:
        """ Normal """
        terms = [self.log_prior_msrmnt,
                 self.log_likelihood]
        return reduce(lambda x, y: x + y(params), [0] + terms)


    def log_joint(self, params) -> float:
        """Joint distribution of parameters and data"""
        terms = [self.log_likelihood,
                 self.log_prior_struct,
                 self.log_prior_msrmnt]
        return reduce(lambda x, y: x + y(params), [0] + terms)


    def prior_params_psi(self, params):
        # for Psi matrix
        ms = self.get_matrices(params)
        psi_dim = ms['Psi'].shape[0]  # dimention of psi matrix
        p_psi_df = self.m_profiles.shape[0]  # number of samples
        p_psi_cov = ms['Psi'] * (p_psi_df - psi_dim - 1)
        return p_psi_df, p_psi_cov


    def prior_params_beta(self, params, param_pos):
        beta_init = [params[i] for i, p in param_pos.items() if p[0] == 'Beta']
        p_beta_mean = np.mean(beta_init)
        p_beta_cov = np.var(beta_init) + 1
        return p_beta_mean, p_beta_cov


    def prior_params_theta(self, params, param_pos):
        theta_init = [params[i] for i, p in param_pos.items() if p[0] == 'Theta']
        p_theta_alpha = self.m_profiles.shape[0]/2
        p_theta_beta = np.median(theta_init) * (p_theta_alpha - 1)
        return p_theta_alpha, p_theta_beta

    def prior_params_lambda(self, params, param_pos):
        lambda_init = [params[i] for i, p in param_pos.items() if p[0] == 'Lambda']
        p_lambda_mean = np.mean(lambda_init)
        p_lambda_cov = np.var(lambda_init) + 1
        return p_lambda_mean, p_lambda_cov