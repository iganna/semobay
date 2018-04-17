from sem_model import SEMModel, SEMData
from sem_opt_classic import SEMOptClassic
from sem_opt_bayes import SEMOptBayes
from sem_inspector import inspect, inspect_mx
from sem_opt_phylo import SEMModelNode
import numpy as np
from scipy.optimize import minimize


from scipy.stats import wishart


estimator = 'MLW'
# estimator = 'MLN'  # ANNA: This method is toooooooo time-consuming now
# estimator = 'GLS'
# estimator = 'ULS'
# estimator = 'MLSkewed'



# path_pref = 'schiza/'
# file_model = path_pref + 'mod_akt_single_cut.txt'
# file_data = path_pref + 'exprs_control.txt'

path_model = 'data/'
path_data = 'data/'
path_res = 'res/'
file_model = 'mod01_full.txt'
file_data = 'example01.txt'

#
# path_model = 'phylogeny/'
# path_data = 'phylogeny/'
# file_data = 'NEU.txt'
# file_model = 'mod_blood.txt'



mod = SEMModel(path_model + file_model)
data = SEMData(mod, path_data + file_data)
mod.load_initial_dataset(data)

reg_lambda = 0.075

opt_classic = SEMOptClassic(mod, data, estimator, regularization='lasso')

with open(path_res + file_model[:-4] + '_before_anna.txt', 'w') as f:
    inspect(mod, opt_classic, f)


opt_classic.optimize(alpha=reg_lambda)


with open(path_res + file_model[:-4] + '_new' + estimator +
                  '_lambda' + str(reg_lambda) + '.txt', 'w') as f:
    inspect(mod, opt_classic, f)

# print(opt_classic.loss_func(opt_classic, mod.param_val))
print(opt_classic.loss_func(opt_classic, opt_classic.params))

# =========================================
opt_classic = SEMOptClassic(mod, data, estimator, regularization='lasso')
opt_classic.optimize()
regul_chain = np.array([opt_classic.params])
for reg_lambda in [x/5 for x in range(100)]:
    print(reg_lambda)
    # opt_classic = SEMOptClassic(mod, data, estimator, regularization='lasso')
    opt_classic.optimize(alpha=reg_lambda)
    regul_chain = np.append(regul_chain, [opt_classic.params], axis=0)

np.savetxt(path_res + 'regul_params.txt', regul_chain, '%.3f')
# =========================================

bayes_estimator = 'EmpBayes'
# bayes_estimator = 'Likelihood'


opt_bayes = SEMOptBayes(mod, data, opt_classic.params, bayes_estimator)
opt_bayes.optimize()
opt_bayes.params = np.median(opt_bayes.param_chain, axis=0)

with open(path_res + file_model[:-4] + '_after_bayes' + estimator + '.txt', 'w') as f:
    inspect(mod, opt_bayes, f)


np.savetxt(path_res + 'chain.txt', opt_bayes.param_chain, '%.3f')

