from sem_model_full import SEMModelFull


path_model = 'full_model/'
file_model = 'mod_gp01.txt'


mod = SEMModelFull(path_model + file_model)

model = mod.model
sem_op = mod.sem_op