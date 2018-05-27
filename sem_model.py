import numpy as np
from sem_percer import SEMParser
from scipy.stats import linregress
from pandas import read_csv
from sys import stdout
import itertools as it
import os

#TODO add none to not set parameters

class SEMData:
    def __init__(self, sem, file_name, center=True):
        """
        This function ...
        :param sem:
        :param file_name:
        :param center:
        :return:
        """

        # TODO assert for file existence

        self.name = os.path.basename(file_name[:-4])
        self.d_vars = sem.d_vars['D_MPart']
        self.m_profiles = self.get_profiles(sem, file_name, center)

        if self.m_profiles.shape[1] < len(self.d_vars):
            # WARNING
            raise ValueError('This dataset is not suitable for the SEM model')

        self.m_cov = np.cov(self.m_profiles, rowvar=False, bias=True)

    @staticmethod
    def get_profiles(sem, file_name, center=True):
        """
        Loads data from pandas-compataible file, adjusts it for sem's model
           and returns a numpy array
        :param sem:
        :param file_name:
        :param center:
        :return:
        """
        sep = ','
        data = read_csv(file_name, sep=sep)
        if min(data.shape) == 1:
            sep = '\t'
            data = read_csv(file_name, sep=sep)
        if min(data.shape) == 1:
            raise ValueError('Dataset is of vector form')
        if 'Unnamed: 0' in list(data):
            data = read_csv(file_name, sep=sep, index_col=0)

        try:
            data = (data[sem.d_vars['MPart']]).as_matrix()
        except:

            data = data.transpose()
            try:
                data = (data[sem.d_vars['MPart']]).as_matrix()
            except:
                print('bad')


        if center:
            data -= data.mean(axis=0)
            return data

    def in_line_with_mod(self, mod):
        """
        Check whether dataset and model are in agreement
        :param mod:
        :return: TRUE of FALSE
        """
        pass


class SEMModel:

    def __init__(self, file_model, diag_psi=False):

        # TODO assert
        model_descr = None
        with open(file_model, 'r') as f:
            model_descr = f.read()
        # TODO add asserts for model_descr

        sem_parser = SEMParser()
        self.sem_op = sem_parser.operations

        # REMOVE
        # sem_op = sem_parser.operations
        # model = sem_parser.parse(model_descr)
        # d_vars = SEMModel.classify_vars(model, sem_op)

        self.model = sem_parser.parse(model_descr)
        self.d_vars = self.classify_vars(self.model, self.sem_op)

        # Set matrices and parameters
        self.diag_psi = diag_psi
        self.n_param = 0
        self.param_pos = {}
        self.param_val = []  # initial values of parameters

        self.matrices = dict()
        self.matrices['Beta'] = self.set_beta()
        self.matrices['Lambda'] = self.set_lambda()
        self.matrices['Theta'] = self.set_theta()
        self.matrices['Psi'] = self.set_psi()

    # # TODO
    # def get_params_from_matrices():
    #     params = np.zeros(self.n_params)
    #     matrices = (self.m_beta, self.m_lambda, self.m_theta, self.m_psi)
    #     for mx, positions in matrices:
    #         for i, pos in positions.items():
    #             params[i] = mx[pos[0][0], pos[0][1]]
    #     return params

    @staticmethod
    def classify_vars(model, sem_op):
        """


        ANNA: variable are classified into categories:
        LatExo, LatEndo, ObsExo, ObsEndo, Manifest, SPart, MPart


        :param model:
        :return:
        """
        acc = {}
        vars_all = {v for v in model}

        # Latent variables
        vars_lat = {v for v in vars_all if model[v][sem_op.MEASUREMENT]}
        # Manifest variables
        vars_menif = {v for latent in vars_lat for v in model[latent][sem_op.MEASUREMENT]}
        # Onserved variables
        vars_obs = vars_all - vars_lat - vars_menif

        # Endogenous variables
        vars_endo = {v for v in vars_all if model[v][sem_op.REGRESSION]}
        # Exogeneous
        vars_exo = (vars_lat | vars_obs) - vars_endo

        vars_upstream = {}
        for v in vars_all:
            if model[v][sem_op.REGRESSION]:
                vars_upstream |= model[v][sem_op.REGRESSION].keys()
        vars_output = vars_endo - vars_upstream

        acc['LatExo'] = sorted(list(vars_lat & vars_exo))
        acc['LatEndo'] = sorted(list(vars_lat & vars_endo))
        acc['ObsExo'] = sorted(list(vars_obs & vars_exo))
        acc['ObsEndo'] = sorted(list(vars_obs & vars_endo))
        acc['Manifest'] = sorted(list(vars_menif))
        acc['Output'] = list(vars_output - set(acc['Manifest']))

        # DO NOT SORT AGAIN
        acc['Lat'] = acc['LatExo'] + acc['LatEndo']
        acc['SPart'] = acc['LatExo'] + acc['LatEndo'] + acc['ObsExo'] + acc['ObsEndo']
        acc['MPart'] = acc['Manifest'] + acc['ObsExo'] + acc['ObsEndo']

        acc['D_First_Manif'] = {latent:[*model[latent][sem_op.MEASUREMENT]][0] for latent in acc['Lat']}

        # Dictionaries for Structural and Measurement parts to fix the order of variables
        acc['D_SPart'] = {v: i for i, v in enumerate(acc['SPart'])}
        acc['D_MPart'] = {v: i for i, v in enumerate(acc['MPart'])}

        acc['All'] = acc['LatExo'] + acc['LatEndo'] + \
                     acc['ObsExo'] + acc['ObsEndo'] + acc['Manifest']
        return acc

    def add_parameter(self, param_type, pos1, pos2=None):
        """
        Add new parameters
        :param param_type:
        :param pos1:
        :param pos2:
        :return:
        """
        self.param_pos[self.n_param] = (param_type, pos1, pos2)
        self.param_val.append(0)
        self.n_param += 1

    def set_beta(self):
        """

        :param vrbls:
        :param model:
        :param n_params:
        :return:
        """
        # variables in structural part of SEM
        v_spart = self.d_vars['SPart']
        d_spart = self.d_vars['D_SPart']

        # create Beta matrix with indicators of parameters
        n_spart = len(v_spart)
        m_beta = np.zeros((n_spart, n_spart))

        for v1, v2 in it.permutations(v_spart, 2):
            if v2 in self.model[v1][self.sem_op.REGRESSION]:
                self.add_parameter('Beta', d_spart[v1], d_spart[v2])


        return m_beta

    def set_lambda(self):
        """

        :param d_vars:
        :param model:
        :param n_params:
        :return:
        """
        # variables in structural part of SEM
        v_spart = self.d_vars['SPart']
        v_mpart = self.d_vars['MPart']
        v_latent = self.d_vars['Lat']
        d_spart = self.d_vars['D_SPart']
        d_mpart = self.d_vars['D_MPart']
        d_fisrt_manif = self.d_vars['D_First_Manif']
        n_spart = len(v_spart)
        n_mpart = len(v_mpart)

        # create Lambda matrix with indicators of parameters
        m_lambda = np.zeros((n_mpart, n_spart))

        # Define fixed_to-one parameters and parameters for estimation
        for v2 in v_latent:

            for v1 in self.model[v2][self.sem_op.MEASUREMENT]:
                if v1 is d_fisrt_manif[v2]:  # for the first - set 1
                    m_lambda[d_mpart[v1], d_spart[v2]] = 1
                else:
                    self.add_parameter('Lambda', d_mpart[v1], d_spart[v2])

        # Set the lower-right block of matrix to identity (diagonal)
        v_sobserved = self.d_vars['ObsExo'] + self.d_vars['ObsEndo']
        for v in v_sobserved:
            m_lambda[d_mpart[v], d_spart[v]] = 1
        return m_lambda

    def set_theta(self):
        """
        Covariance matrix of errors in measurement part

        :param d_vars:
        :param model:
        :param n_params:
        :return:
        """
        v_mpart = self.d_vars['MPart']
        n_mpart = len(v_mpart)
        d_mpart = self.d_vars['D_MPart']
        v_manifest = self.d_vars['Manifest']

        m_theta = np.zeros((n_mpart, n_mpart))
        params = {}
        for v in v_manifest:
            self.add_parameter('Theta', d_mpart[v], d_mpart[v])

        # # TODO
        # # Covariances between variables set in model
        # # DO NOT FORGET THE DIAGONAL SIMILARITY
        # for v1, v2 in product(v_mpart, v_mpart):
        #     if v2 in model[v1][Operations.REGRESSION]:
        #         params[n_params] = (d_mpart[v1], d_mpart[v2])
        #         n_params += 1

        return m_theta

    def set_psi(self):
        """
        The true is :
        All latent Variables: param var - started with 0.5
        Endogenous - all covariances - parameters
        ExogenLat - all covariances - parameters


        # Endogenous - parameters between every two variables
        #     Endogenous Latent -  param var - started with 0.05
        #                          param cov - started with zeros
        #     Endogenous Observed - param var - sampled var divided by 2;
        #                           param cov between "out"-variables - started with zeros
        #     param cov between EnDoLat and EndoObs - started with zeros
        #
        # Exogenous
        #     Exogenous Latent - param var - 0.5
        #                        param cov - zero
        #     Exogenous Observed - fixed var - sampled var
        #                          fixed cov - fixed to zero

        :param d_vars:
        :param model:
        :param n_params:
        :return:
        """

        d_spart = self.d_vars['D_SPart']
        n_spart = len(d_spart)

        m_psi = np.zeros((n_spart, n_spart))

        # Set Variances for Latent (Endo and Exo) variables as parameters with 0.05
        v_latent = self.d_vars['Lat']
        for v in v_latent:
            self.add_parameter('Psi', d_spart[v], d_spart[v])

        # Define parameters of variances for Endogenous observed variables
        v_obs_endo = self.d_vars['ObsEndo']
        for v in v_obs_endo:
            self.add_parameter('Psi', d_spart[v], d_spart[v])

        if self.diag_psi is True:
            return m_psi

        # Define parameters of covariances within a group of Exogenous Latent Variables
        v_exo_lat = self.d_vars['LatExo']
        for v1, v2 in it.combinations(v_exo_lat, 2):
            self.add_parameter('Psi', d_spart[v1], d_spart[v2])

        # Define parameters of covariances for all output-endogenous variables
        v_endo = self.d_vars['Output']
        for v1, v2 in it.combinations(v_endo, 2):
            self.add_parameter('Psi', d_spart[v1], d_spart[v2])

        # TODO
        # ADDITIONAL COVARIANCES FROM MODEL
        return m_psi

    def update_all(self, params=None):
        """
        Set New parameters or Update matrixes from parameters
        :param params:
        :return:
        """
        # TODO LEN of parameters
        if params is not None:
            self.check_params(params)
            # self.param_val = params
        else:
            params = self.param_val

        # ANNA print('updated', params)
        for i, position in self.param_pos.items():
            mx_type, pos1, pos2 = position
            if mx_type in {'Beta', 'Lambda'}:
                self.matrices[mx_type][pos1, pos2] = params[i]
            elif mx_type in {'Psi', 'Theta'}:  # Symmetric matrices
                self.matrices[mx_type][pos1, pos2] = params[i]
                self.matrices[mx_type][pos2, pos1] = params[i]

    def load_initial_dataset(self, data: SEMData):
        """
        Set Initial values into Matrices
        :param data:
        :return:
        """

        # TODO check whether the dataset and the model are in agreement

        m_cov = data.m_cov
        m_profiles = data.m_profiles
        d_spart = self.d_vars['D_SPart']
        d_mpart = self.d_vars['D_MPart']
        v_spart = self.d_vars['SPart']
        v_mpart = self.d_vars['MPart']
        v_obsexo = self.d_vars['ObsExo']
        v_lat = self.d_vars['Lat']
        d_first_manif = self.d_vars['D_First_Manif']

        # Matrix Beta - regression coefficients
        for i, position in self.param_pos.items():
            mx_type, pos1, pos2 = position
            if mx_type != 'Beta':
                continue

            # if v_spart[pos1] in v_lat:
            #     profile1 = m_profiles[:, d_mpart[d_first_manif[v_spart[pos1]]]]
            # else:
            #     profile1 = m_profiles[:, d_mpart[v_spart[pos1]]]
            #
            # if v_spart[pos2] in v_lat:
            #     profile2 = m_profiles[:, d_mpart[d_first_manif[v_spart[pos2]]]]
            # else:
            #     profile2 = m_profiles[:, d_mpart[v_spart[pos2]]]

            # lin_reg = linregress(profile2, profile1)
            # self.param_val[i] = lin_reg.slope
            self.param_val[i] = 0

        # Matrix Lambda - regression coefficients
        # first pos1 from MPart
        # second pos2 - from SPart
        for i, position in self.param_pos.items():
            mx_type, pos1, pos2 = position
            if mx_type != 'Lambda':
                continue

            # first - is measured
            profile1 = m_profiles[:, pos1]
            # second is always latent
            profile2 = m_profiles[:, d_mpart[d_first_manif[v_spart[pos2]]]]

            lin_reg = linregress(profile2, profile1)
            self.param_val[i] = lin_reg.slope

        # Matrix Psi - covariances from structural part not for latent variables
        for i, position in self.param_pos.items():
            mx_type, pos1, pos2 = position
            if mx_type != 'Psi':
                continue

            if pos1 != pos2:
                continue

            if v_spart[pos1] in v_lat:
                self.param_val[i] = 0.05
                continue

            # Translation from structural part into measurement
            pos1 = pos2 = d_mpart[v_spart[pos1]]

            # only diagonal variance
            self.param_val[i] = m_cov[pos1][pos2] / 2



        # Matrix Theta - covariances from measurement part
        for i, position in self.param_pos.items():
            mx_type, pos1, pos2 = position
            if mx_type != 'Theta':
                continue
            self.param_val[i] = m_cov[pos1][pos2] / 2

        # Psi matrix :  set fixed variances for exogenous observed variables
        for v1, v2 in it.product(v_obsexo, repeat=2):  # all symmetric variants
            pos1 = d_spart[v1]
            pos2 = d_spart[v2]

            # Translation from structural part into measurement
            pos_cov1 = d_mpart[v1]
            pos_cov2 = d_mpart[v2]

            self.matrices['Psi'][pos1, pos2] = m_cov[pos_cov1, pos_cov2]

        # DO NOT Update matrices

    def get_matrices(self, params=None, mx_type='All'):
        """
        Matrices are updated by default
        :param params:
        :return:
        """
        if params is not None:
            self.check_params(params)

        self.update_all(params)


        # The returned matrices are updated
        if mx_type is 'All':
            return self.matrices
        if mx_type in self.matrices.keys():
            return self.matrices[mx_type]

    def get_bounds(self):
        mx_to_fix = ['Theta', 'Psi']
        bounds = []
        for i, position in self.param_pos.items():
            mx_type, pos1, pos2 = position
            if mx_type in mx_to_fix:
                bounds.append((0, None))
            else:
                bounds.append((None, None))
        return bounds

    def check_params(self, params):
        """
        Chechs parameters
        :param params:
        :return:
        """
        if len(params) != self.n_param:
            print(len(params), self.n_param)
            raise ValueError('Length of parameters is not valid')
        if not all(isinstance(p, (int, float)) for p in params):
            raise ValueError('Invalid values')

    def check_covariances(self, output=stdout):
        """
        Checks matricies' values to be A-OK.
        :param output:
        :return:
        """
        if np.any(np.diag(self.m_theta[0]) < 0) or np.any(np.diag(self.m_psi[0]) < 0):
            print("Warning: some of variances are negative.", file=output)
            raise ValueError("Warning: some of variances are negative.")
        if np.any(np.linalg.eigvals(self.m_theta[0]) < 0):
            print("Warning: theta matrix is not positive definite.", file=output)
            raise ValueError("Warning: theta matrix is not positive definite")
        if np.any(np.linalg.eigvals(self.m_psi[0]) < 0):
            print("Warning: psi matrix is not positive definite.", file=output)
            raise ValueError("Warning: psi matrix is not positive definite.")