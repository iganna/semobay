from sem_model import SEMModel, SEMData
from sem_opt_classic import SEMOptClassic
from sem_opt_bayes import SEMOptBayes
from sem_inspector import inspect, inspect_mx
from sem_opt_phylo import SEMModelNode
import numpy as np
from scipy.optimize import minimize

from sem_stats import gather_pvals
from sem_regul import  SEMCrossVal

from scipy.stats import wishart


estimator = 'MLW'
# estimator = 'MLN'  # ANNA: This method is toooooooo time-consuming now
# estimator = 'GLS'
# estimator = 'ULS'
# estimator = 'MLSkewed'



# path_pref = 'schiza/'
# file_model = path_pref + 'mod_akt_single_cut.txt'
# file_data = path_pref + 'exprs_control.txt'

# path_model = 'data/'
# path_data = 'data/'
# path_res = 'res/'
# file_model = 'mod01_full.txt'
# file_data = 'example01.txt'


path_model = 'gpsem/'
path_data = 'gpsem/'
path_res = 'gpsem/'
file_model = 'mod_genphen.txt'
file_data = 'pca8.txt'
file_data = 'phen.txt'

# file_model = 'mod_gp01.txt'


#
# path_model = 'phylogeny/'
# path_data = 'phylogeny/'
# file_data = 'NEU.txt'
# file_model = 'mod_blood.txt'

mod = SEMModel(path_model + file_model, diag_psi=True)
data = SEMData(mod, path_data + file_data)
mod.load_initial_dataset(data)

reg_alpha = 0

# # =========================================
#
# opt_classic = SEMOptClassic(mod, data, estimator, 'l2')
#
# opt_classic.optimize(alpha=reg_alpha)
#
# with open(path_res + file_model[:-4] + '_georg' + estimator +
#                   '_lambda' + str(reg_alpha) + '_best2.txt', 'w') as f:
#     inspect(mod, opt_classic, f)
#
# # print(opt_classic.loss_func(opt_classic, mod.param_val))
# print(opt_classic.loss_func(opt_classic, opt_classic.params))
#
# # =========================================



opt_classic = SEMOptClassic(mod, data, estimator, regularization='l2')
# opt_classic.fix_matrix({'Lambda'})
opt_classic.optimize(alpha=10 ** (-10))
opt_classic.fix_matrix({'Psi', 'Theta', 'Lambda'})
params_init = np.array(opt_classic.params)

# np.savetxt(path_res + 'params_init_x3.txt', opt_classic.params, '%.10f')


opt_cv = SEMCrossVal(mod, data, estimator, 4)
opt_cv.fix_matrix({'Psi', 'Theta', 'Lambda'})


# reg_degrees = [x/20 - 3 for x in range(200)]
reg_degrees = [x/10 - 5 for x in range(100)]

reg_range = [10 ** d for d in reg_degrees]



regul_chain = np.array([opt_classic.params])
lld = []
for reg_alpha in reg_range:
    # opt_classic = SEMOptClassic(mod, data, estimator, regularization='l2')
    opt_classic.optimize(alpha=reg_alpha)
    params_init = np.array(opt_classic.params)
    pvals = gather_pvals(opt_classic, data)

    pvalth_thresh_set = [p for id, p in enumerate(pvals)
                         if (0 < p < 1) and
                         (opt_classic.param_pos[id][0] == 'Beta')]
    pvalth_thresh_set.sort()
    if len(pvalth_thresh_set) >= 5:
        pvalth_thresh = pvalth_thresh_set[len(pvalth_thresh_set) - 5]
    elif len(pvalth_thresh_set) == 1:
        pvalth_thresh = pvalth_thresh_set[0]
    else:
        pvalth_thresh = 0
    p_val_flag = False
    for id, p in enumerate(pvals):
        if (p > 0.5) and (p >= pvalth_thresh) \
                and (opt_classic.param_pos[id][0] == 'Beta'):
            opt_classic.fix_param_zero(id)
            opt_cv.fix_param_zero(id)
            p_val_flag = True

    if p_val_flag:
        log_likelihood = opt_cv.cv_likelihood(params_init)
        lld += [log_likelihood]
        print(log_likelihood)

        mx = opt_classic.calculate_sigma(opt_classic.params)
        x = opt_classic.ml_norm_log_likelihood(mx, opt_classic.m_profiles)
    else:
        lld += [0]
    regul_chain = np.append(regul_chain, [opt_classic.params], axis=0)

np.savetxt(path_res + 'regul_params12.txt', regul_chain, '%.3f')


params = regul_chain[56]

m_beta = opt_classic.get_matrices(params, 'Beta')
np.savetxt(path_res + 'beta_12.txt', m_beta, '%.3f')

np.savetxt(path_res + 'log_likelihood11.txt', lld, '%.3f')


g = opt_classic.compose_gradient_function(0)
g(opt_classic.params)

np.savetxt(path_res + 'lambda12.txt', opt_classic.get_matrices(
    opt_classic.params, 'Lambda'), '%.3f')
np.savetxt(path_res + 'beta9.txt', opt_classic.get_matrices(
    opt_classic.params, 'Beta'), '%.3f')

# =========================================
mx = opt_classic.calculate_sigma(opt_classic.params)
opt_classic.ml_norm_log_likelihood(mx, opt_classic.m_profiles)

# =========================================

bayes_estimator = 'EmpBayes'
# bayes_estimator = 'Likelihood'


opt_bayes = SEMOptBayes(mod, data, opt_classic.params, bayes_estimator)
opt_bayes.optimize()
opt_bayes.params = np.median(opt_bayes.param_chain, axis=0)

with open(path_res + file_model[:-4] + '_after_bayes' + estimator + '.txt', 'w') as f:
    inspect(mod, opt_bayes, f)


np.savetxt(path_res + 'chain.txt', opt_bayes.param_chain, '%.3f')

