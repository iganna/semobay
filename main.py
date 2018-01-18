
from semopy_model import SEMModel, SEMData
from semopy_optimiser import SEMOptimiser
from semopy_inspector import inspect, inspect_mx
import numpy as np
from scipy.optimize import minimize


# estimator = 'MLW'
# estimator = 'MLN'  # ANNA: This method is toooooooo time-consuming now
# estimator = 'GLS'
# estimator = 'ULS'
estimator = 'MLSkewed'



path_pref = 'schiza/'
file_model = path_pref + 'mod_akt_single_cut.txt'
file_data = path_pref + 'exprs_control.txt'

# path_pref = 'data/'
# file_model = path_pref + 'mod06.txt'
# file_data = path_pref + 'example06.txt'


mod = SEMModel(file_model)
data = SEMData(mod, file_data)
mod.load_initial_dataset(data)

# Pre optimisation
# sem_optimiser0 = SEMOptimiser(mod, data, 'ULS')
# sem_optimiser0.optimize()

sem_optimiser = SEMOptimiser(mod, data, estimator, 'L2')

with open(file_model[:-4] + '_before_anna.txt', 'w') as f:
    inspect(mod, sem_optimiser, f)

sem_optimiser.optimize()

with open(file_model[:-4] + '_after_anna_new' + estimator + '.txt', 'w') as f:
    inspect(mod, sem_optimiser, f)
# for mx_name in mod.matrices.keys():
#     with open(file_model[:-4] + '_mx_' + mx_name + '.txt', 'w') as f:
#         inspect_mx(mx_name, mod, sem_optimiser, f)


#semPlot(sem)