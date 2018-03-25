from sem_model import SEMData, SEMModel
from sem_opt_phylo import SEMModelNode, SEMTree, SEMOptPhylo
from sem_inspector_phylo import inspect
import numpy as np

path_model = 'phylogeny/'
path_data = 'phylogeny/'
path_tree = 'phylogeny/'

path_res = path_data + 'res/'

file_model = 'mod_blood.txt'
files_data = ['CD4.txt', 'CD8.txt', 'MON.txt', 'NEU.txt']
file_tree = 'tree_init.nwk'


mod_leaf = SEMModel(path_model + file_model)
mod_node = SEMModelNode(path_model + file_model)

dataset = [SEMData(mod_leaf, path_data + f) for f in files_data]
mod_node.load_initial_dataset(dataset[0])

tree = SEMTree(dataset, path_tree + file_tree)

opt_phylo = SEMOptPhylo(mod_node, dataset, tree)

opt_phylo.optimise()

opt_phylo.param_val = opt_phylo.param_chain[-1]

for node_name in opt_phylo.tree.keys():
    with open(path_res + file_model[:-4] + 'node_' + node_name + '_2.txt', 'w') as f:
        inspect(mod_node, opt_phylo, node_name, f)


np.savetxt(path_res + 'chain6.txt', opt_phylo.param_chain, '%.3f')


with open(path_res + file_model[:-4] + '_all_nodes' + '.txt', 'w') as f:
    for node_name in opt_phylo.tree.keys():
        params_node = opt_phylo.get_node_params(opt_phylo.param_val, node_name)
        params_beta = [params_node[i] for i, k in mod_node.param_pos.items()
                       if k[0] == 'Beta']
        print(node_name, params_beta, file=f)



