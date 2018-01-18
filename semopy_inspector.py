from pandas import DataFrame, set_option, reset_option
from semopy_model import SEMModel
from semopy_optimiser import SEMOptimiser
from graphviz import Digraph
from sys import stdout
from numpy.linalg import norm


def inspect(sem: SEMModel, sem_optimiser: SEMOptimiser,  output=stdout):
    '''Outputs model's matricies with their current parameters assigned their resepctive values.'''

    def print_matrix(matrixName, matrix, colHeaders, rowHeaders, output):
        lenX = len(rowHeaders)
        lenY = len(colHeaders)
        print(matrixName, file=output)
        print("Shape:", matrix.shape, file=output)
        print(DataFrame(matrix[:lenX, :lenY], columns=colHeaders, index=rowHeaders).round(3), file=output)

    set_option('expand_frame_repr', False)
    v_all = sem.d_vars['All']
    v_obs = sem.d_vars['ObsEndo'] + sem.d_vars['ObsExo'] + sem.d_vars['Manifest']
    v_spart = sem.d_vars['SPart']
    v_mpart = sem.d_vars['MPart']
    v_latent = sem.d_vars['Lat']
    v_endo = sem.d_vars['LatEndo'] + sem.d_vars['ObsEndo']
    v_exo = sem.d_vars['LatExo'] + sem.d_vars['ObsExo']

    print('All variables:', v_all, file=output)
    print('Observable variables:', v_obs, file=output)
    print('Structural part:', v_spart, file=output)
    print('Measurement part:', v_mpart, file=output)
    print('Latent varaibles:', v_latent, file=output)
    print('Endogenous variables:', v_endo, file=output)
    print('Exogenous variables:', v_exo, file=output)

    if sem_optimiser.estimator == 'MLSkewed':
        print('Skewness ', sem_optimiser.add_params, file=output)

    matrices = sem.get_matrices(sem_optimiser.params)
    print_matrix('Beta', matrices['Beta'], v_spart, v_spart, output)
    print_matrix('Lambda', matrices['Lambda'], v_spart, v_mpart, output)
    print_matrix('Psi', matrices['Psi'], v_spart, v_spart, output)
    print_matrix('Theta', matrices['Theta'], v_mpart, v_mpart, output)

    m_cov = sem_optimiser.m_cov
    if m_cov is not None:
        m_sigma = sem_optimiser.calculate_sigma()
        print_matrix("Empirical covariance matrix:", m_cov, v_mpart, v_mpart, output)
        print_matrix("Model-implied covariance matrix:", m_sigma, v_mpart, v_mpart, output)
        print("Euclidian difference between them:", norm(m_cov - m_sigma), file=output)
    reset_option('expand_frame_repr')



def inspect_mx(mx_name, sem: SEMModel, sem_optimiser: SEMOptimiser,  output=stdout):
    '''Outputs model's matricies with their current parameters assigned their resepctive values.'''

    def print_matrix(matrixName, matrix, colHeaders, rowHeaders, output):
        lenX = len(rowHeaders)
        lenY = len(colHeaders)
        print(DataFrame(matrix[:lenX, :lenY], columns=colHeaders, index=rowHeaders).round(3), file=output)

    set_option('expand_frame_repr', False)
    v_all = sem.d_vars['All']
    v_obs = sem.d_vars['ObsEndo'] + sem.d_vars['ObsExo'] + sem.d_vars['Manifest']
    v_spart = sem.d_vars['SPart']
    v_mpart = sem.d_vars['MPart']
    v_latent = sem.d_vars['Lat']
    v_endo = sem.d_vars['LatEndo'] + sem.d_vars['ObsEndo']
    v_exo = sem.d_vars['LatExo'] + sem.d_vars['ObsExo']



    matrices = sem.get_matrices(sem_optimiser.params)

    if mx_name == 'Beta':
        print_matrix('Beta', matrices['Beta'], v_spart, v_spart, output)
    elif mx_name == 'Lambda':
        print_matrix('Lambda', matrices['Lambda'], v_spart, v_mpart, output)
    elif mx_name == 'Psi':
        print_matrix('Psi', matrices['Psi'], v_spart, v_spart, output)
    elif mx_name == 'Theta':
        print_matrix('Theta', matrices['Theta'], v_mpart, v_mpart, output)





#
# def semPlot(sem, labelEdges=False, covariances=False, filename=None):
#     g = Digraph(engine='dot', format='jpg')
#     g.edge_attr.update({'fontsize': '8'})
#     sp = sem.m_model.m_vars['SPart']
#     mp = sem.m_model.m_vars['MPart']
#     lv = sem.m_model.m_vars['Latent']
#     nlv = sorted(list(set(sp) - set(lv)) + mp)
#     for v in nlv:
#         g.node(v, shape='box', color='red')
#     for v in mp:
#         g.node(v, shape='box', color='black')
#     for v in lv:
#         g.node(v, shape='oval', color='red')
#     for v in sem.m_model.m_model:
#         for rvalue in sem.m_model.m_model[v][Operations.MEASUREMENT]:
#             if labelEdges:
#                 val = round(sem.m_model.get_value_from_operation(Operations.MEASUREMENT, v, rvalue), 3)
#             else:
#                 val = ''
#             g.edge(v, rvalue, label=str(val))
#         for rvalue in sem.m_model.m_model[v][Operations.REGRESSION]:
#             if labelEdges:
#                 val = round(sem.m_model.get_value_from_operation(Operations.REGRESSION, v, rvalue), 3)
#             else:
#                 val = ''
#             g.edge(rvalue, v, label=str(val))
#         if covariances:
#             for rvalue in sem.m_model.m_model[v][Operations.COVARIANCE]:
#                 if labelEdges:
#                     val = round(sem.m_model.get_value_from_operation(Operations.COVARIANCE, v, rvalue), 3)
#                 else:
#                     val = ''
#                 g.edge(v, rvalue, dir='both', style='dashed', constraint='false', label=str(val))
#     if filename is None:
#         g.render(view=True)
#     else:
#         g.render(filename)
#
