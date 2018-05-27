from sem_model_full import SEMModelFull, SEMmx
from sem_model_full import SEMDataFull
from sem_opt_bfull import SEMOptBayesFull

from sem_model import SEMModel, SEMData
from sem_opt_classic import SEMOptClassic

import numpy as np
import scipy.stats as st

# TODO ALL CATEGORIAL DATA SHOULD START FROM 0. then 1 etc

path_model = 'gpsem/'
file_model = 'gpsem_mod.txt'
file_model = 'gpsem_mod_03.txt'
file_data = 'gpsem_test01.txt'


#
# mod_classic = SEMModel(path_model + file_model, )
# data_classic = SEMData(mod_classic, path_model + file_data)
# mod_classic.load_initial_dataset(data_classic)
# opt_classic = SEMOptClassic(mod_classic, data_classic, 'MLW')
# opt_classic.optimize()
#
# list(zip(opt_classic.param_pos, opt_classic.params))


mod = SEMModelFull(path_model + file_model)
data = SEMDataFull(mod, path_model + file_data)
mod.load_dataset(data)

param_prior = [3.4, 2.6, 3.0, 1, 0.36, 0.1, 15, 0.72, 1.69, 0.24,
               0.25, 0.25, 0.25]
opt = SEMOptBayesFull(mod, data, param_prior)
# self = opt

# param_lavaan = [1.402, 2.017, 4.502, 1.278, 3.403, 0.370, 1.394, 0.659,
#                 1.766, 3.085, 1., 3.221, 1., 0.113, 0.382, 0.115, 1., 0.313,
#                 0.270, 0.358, 6.011, 5.715, 0.887, 1.137, 0.946, 1.954,
#                 0.079, 0.240, 1.551, 0.250, 0.250, 0.250, 0.250, 0.250,
#                 0.250, 0.250]
# opt.param_val = param_lavaan



mcmc = opt.optimise()
np.savetxt(path_model + 'mcmc3_4.txt', mcmc)


opt.get_matrix(SEMmx.BETA, opt.param_val)
opt.get_matrix(SEMmx.GAMMA, opt.param_val)
opt.get_matrix(SEMmx.PI, opt.param_val)
opt.get_matrix(SEMmx.LAMBDA, opt.param_val)


#
# x = opt.get_matrix(SEMmx.MPART, [p+1 for p in opt.param_val])
#
# np.savetxt(path_model + 'tmp.txt', x)
#
#
# opt.get_matrix(SEMmx.KAPPA, [p+1 for p in opt.param_val])



list(zip(opt.param_val, opt.param_pos))
