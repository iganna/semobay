from pandas import DataFrame, set_option, reset_option
from sem_model import SEMModel
from sem_opt_phylo import SEMOptPhylo
from sys import stdout
from numpy.linalg import norm


def inspect(mod: SEMModel, opt_phylo: SEMOptPhylo, node_name, output=stdout):
    """ Outputs model's matricies with their current parameters assigned their resepctive values. """

    def print_matrix(matrixName, matrix, colHeaders, rowHeaders, output):
        lenX = len(rowHeaders)
        lenY = len(colHeaders)
        print(matrixName, file=output)
        print('Shape:', matrix.shape, file=output)
        print(DataFrame(matrix[:lenX, :lenY], columns=colHeaders, index=rowHeaders).round(3), file=output)

    set_option('expand_frame_repr', False)
    v_all = mod.d_vars['All']
    v_obs = mod.d_vars['ObsEndo'] + mod.d_vars['ObsExo'] + mod.d_vars['Manifest']
    v_spart = mod.d_vars['SPart']
    v_mpart = mod.d_vars['MPart']
    v_latent = mod.d_vars['Lat']
    v_endo = mod.d_vars['LatEndo'] + mod.d_vars['ObsEndo']
    v_exo = mod.d_vars['LatExo'] + mod.d_vars['ObsExo']

    print('All variables:', v_all, file=output)
    print('Observable variables:', v_obs, file=output)
    print('Structural part:', v_spart, file=output)
    print('Measurement part:', v_mpart, file=output)
    print('Latent varaibles:', v_latent, file=output)
    print('Endogenous variables:', v_endo, file=output)
    print('Exogenous variables:', v_exo, file=output)

    # if sem_optimiser.estimator == 'MLSkewed':
    #     print('Skewness ', sem_optimiser.add_params, file=output)

    params_mod = opt_phylo.get_node_params(opt_phylo.param_val, node_name)
    matrices = opt_phylo.get_matrices(params_mod)
    print_matrix('Beta', matrices['Beta'], v_spart, v_spart, output)
    print_matrix('Lambda', matrices['Lambda'], v_spart, v_mpart, output)
    print_matrix('Psi', matrices['Psi'], v_spart, v_spart, output)
    print_matrix('Theta', matrices['Theta'], v_mpart, v_mpart, output)

    if node_name in opt_phylo.m_cov.keys():
        m_cov = opt_phylo.m_cov[node_name]
    else:
        m_cov = opt_phylo.get_matrices(params_mod, 'Cov')


    m_sigma = opt_phylo.calculate_sigma(params_mod)
    print_matrix("Empirical covariance matrix:", m_cov, v_mpart, v_mpart, output)
    print_matrix("Model-implied covariance matrix:", m_sigma, v_mpart, v_mpart, output)
    print("Euclidian difference between them:", norm(m_cov - m_sigma), file=output)

    reset_option('expand_frame_repr')


