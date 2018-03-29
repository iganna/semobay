import numpy as np
from sem_model import SEMData, SEMModel
from sem_opt_classic import SEMOptClassic
from sem_opt_abc import SEMOptABC
from sem_opt_phylo import SEMOptPhylo, SEMModelNode, SEMTreeNode, SEMTree, Parameter
from ete3 import Tree
from typing import List
from sem_opt_bayes import SEMOptBayes
from itertools import combinations_with_replacement, combinations
from functools import reduce
from scipy.stats import invwishart, invgamma, wishart, norm, uniform, multivariate_normal
from functools import partial



class SEMBranch:

    def __init__(self, node_names, node_attrs, b_length):
        """
        This function creates the
        :param data:
        :param file_name:
        """

        self.node_name1, self.node_name2, self.node_name3 = node_names
        self.node_attr1, self.node_attr2, self.node_attr3 = node_attrs
        self.n_nodes = 4


        self.b_length = b_length
        self.param = b_length/5

        self.set_param(self.param)



    def get_nodes(self, tree):
        """
        Recursive function to get all nodes and distances between them
        :param tree:
        :return:
        """
        if tree.name == '':
            tree.name = 'N' + str(self.n_nodes)
            self.n_nodes += 1
            self.nodes[tree.name] = SEMTreeNode('node')
        else:
            self.nodes[tree.name] = SEMTreeNode('leaf')

        if tree.up is not None:
            name_p = tree.up.name  # parent
            name_c = tree.name  # children
            dist = tree.dist

            self.nodes[name_p].add_dist(name_c, dist)
            self.nodes[name_c].add_dist(name_p, dist)

        for node in tree.children:
            self.get_nodes(node)
        return

    def set_param(self, param_new):
        self.nodes = dict()
        self.param = param_new

        self.nodes[self.node_name1] = SEMTreeNode(self.node_attr1)
        self.nodes[self.node_name1].add_dist('Star', self.param)

        self.nodes[self.node_name2] = SEMTreeNode(self.node_attr2)
        self.nodes[self.node_name2].add_dist('Star', self.b_length - self.param)

        self.nodes[self.node_name3] = SEMTreeNode(self.node_attr3)
        self.nodes[self.node_name3].add_dist('Star', self.b_length - self.param)


        self.nodes['Star'] = SEMTreeNode('node')
        self.nodes['Star'].add_dist(self.node_name3, self.b_length - self.param)
        self.nodes['Star'].add_dist(self.node_name2, self.b_length - self.param)
        self.nodes['Star'].add_dist(self.node_name1, self.param)



class SEMOptBranch:

    def __init__(self,
                 mod_node: SEMModelNode,
                 opt_phylo: SEMOptPhylo,
                 dataset: List[SEMData],
                 node_names, node_attrs,
                 estimator='From_Root'):
        """

        :param mod_leaf:
        :param mod_node:
        :param dataset:
        :param tree:
        """

        # Optimal parameters
        self.opt_param_val = opt_phylo.param_val
        self.opt_param_pos = opt_phylo.param_pos

        # Node Names
        self.node_name1, self.node_name2, self.node_name3 = node_names
        # Length of Branch
        for none_name, d, in opt_phylo.tree[self.node_name1].dist:
            if none_name == self.node_name2:
                self.b_length = d
                break

        # Create tree
        self.tree_init = SEMBranch(node_names, node_attrs, self.b_length)
        self.tree = self.tree_init.nodes

        # Function to pump parameters through matrices
        self.get_matrices = mod_node.get_matrices
        self.get_cov_param = mod_node.get_cov_param
        self.n_param_mod = mod_node.n_param

        # Required Data
        self.m_profiles = {data.name: data.m_profiles for data in dataset
                           if data.name in (self.node_name2, self.node_name3)}
        # Covariance matrix
        self.m_cov = {data.name: data.m_cov for data in dataset
                      if data.name in (self.node_name1, self.node_name2,
                                       self.node_name3)}  #


        # Get prior distributions
        # For this purpose ML-Wishard optimisation must be performed
        self.param_leaf = dict()
        for data in dataset:
            estimator_classic = 'MLW'
            try:
                opt_classic = SEMOptClassic(mod_node, data,
                                            estimator_classic)
                opt_classic.optimize()
            except:
                raise ValueError('SEM models within leaves do not converge')
            self.param_leaf[data.name] = opt_classic.params


        # Create all of the params
        self.param_pos = []
        self.param_val = []
        self.n_params = 0
        self.get_params(self.tree_init, mod_node)

        print(self.param_val[187])

        # Get priors for Beta, Lambda, Psi and Theta matrices
        self.p_psi_df, self.p_psi_cov = self.prior_params_psi()
        self.p_beta_mean, self.p_beta_cov = self.prior_params_coefs('Beta')
        self.p_theta_alpha, self.p_theta_beta = self.prior_params_theta()
        self.p_lambda_mean, self.p_lambda_cov = self.prior_params_coefs('Lambda')
        self.p_tree_alpha, self.p_tree_beta = self.prior_params_tree()

        # Starting values of parameters
        self.set_params(mod_node, dataset)
        print(self.param_val[187])
        # Load optimised parameters
        self.set_optimal_params()
        print(self.param_val[187])

        # Options for the optimisation
        self.param_chain = np.array([self.param_val])
        self.loss_func = self.get_loss_function(estimator)

        print("SEMOptPhylo is successfully created")


    def loss_functions(self) -> dict:
        """
        Create the dictionary of possible functions
        :return:
        """
        tmp_dict = dict()
        tmp_dict['From_Root'] = (('Cov', self.log_post_cov, self.constraint_cov),
                                 ('Beta', self.log_post_beta, self.constraint_sigma),
                                 ('Branch', self.log_post_branch, self.constraint_branch))

        return tmp_dict

    def get_loss_function(self, name):
        loss_dict = self.loss_functions()
        if name in loss_dict.keys():
            return loss_dict[name]
        else:
            raise Exception("SEMOpt_phylo Backend doesn't support loss function {}.".format(name))


    def optimise(self):

        params_init = np.array(self.param_val)
        params_opt = np.array(self.param_val)


        for n_iter in range(200):
            print(n_iter)

            for mx_type, log_prob, constraint_func in self.loss_func:

                if mx_type == 'Cov':  # Common parameters for all nodes
                    node_order = ['Star']
                elif mx_type == 'Branch':
                    node_order = [mx_type]
                    mx_type = ['Star']
                elif mx_type == 'Beta':
                    node_order = self.node_name3 + 'Star'
                else:
                    continue

                for node_name in node_order:
                    # print(node_name, mx_type)
                    params_opt = self.metropolis_hastings(node_name,
                                                          mx_type,
                                                          log_prob,
                                                          constraint_func,
                                                          params_opt)

            self.param_chain = np.append(self.param_chain, [params_opt], axis=0)

        self.param_val = params_opt

        prob_init = self.log_joint(params_init)
        prob_final = self.log_joint(params_opt)
        return prob_init, prob_final


    def metropolis_hastings(self, node_name, mx_type, log_prob, constraint_func, params_opt):
        params_new = np.array(params_opt)

        for pos in self.param_pos:
            if (pos.mx_type not in mx_type) or (pos.node_name not in node_name):
                continue
            # print(node_name, mx_type)
            # print(pos.id_opt)
            p = params_opt[pos.id_opt]

            # Try five times to get a parameter which satisfies the constraint
            for _ in range(5):
                if node_name == 'Branch':
                    p_new = norm.rvs(p, 3, 1)
                else:
                    p_new = norm.rvs(p, 0.05, 1)
                params_new[pos.id_opt] = p_new
                # Constraint
                if constraint_func(params_new, node_name) == 0:
                    break
            # If the required value was not sampled - do not accept it
            if constraint_func(params_new, node_name) < 0:
                params_new[pos.id_opt] = p
                continue


            # Calculate the Metropolis-Hastings statistics
            mh_log_stat = np.exp(log_prob(params_new, node_name) -
                                 log_prob(params_opt, node_name))
            # print(mh_log_stat)
            # print('1', log_prob(params_new, node_name), node_name)
            # print('2', log_prob(params_opt, node_name), node_name)
            # print(node_name, mx_type)

            if (mh_log_stat < uniform.rvs(0, 1, 1)) \
                    or (mh_log_stat == 1):
                # Reject new value
                params_new[pos.id_opt] = p
                if node_name == 'Branch':
                    print('BRANCH', p)
                    self.tree_init.set_param(p)
                    self.tree = self.tree_init.nodes
            else:
                print(node_name, pos.mx_type, mh_log_stat)
                # Accept new value
                params_opt[pos.id_opt] = params_new[pos.id_opt]
                if node_name == 'Branch':
                    print('BRANCH', p_new)
                    self.tree_init.set_param(p_new)
                    self.tree = self.tree_init.nodes

        return params_new


    def get_node_params(self, params_opt, node_name):
        params_mod = np.zeros(self.n_param_mod)
        for p in self.param_pos:
            if p.node_name not in node_name:
                continue
            params_mod[p.id_mod] = params_opt[p.id_opt]
        return params_mod


    def set_node_params(self, params_mod, params_opt, node_name):

        for p in self.param_pos:
            if p.node_name not in node_name:
                continue
            params_opt[p.id_opt] = params_mod[p.id_mod]


    def set_optimal_params(self):

        for p in self.param_pos:
            for opt_p in self.opt_param_pos:
                if (p.node_name == opt_p.node_name) and \
                    (p.mx_type == opt_p.mx_type) and \
                        (p.id_mod == opt_p.id_mod):
                    self.param_val[p.id_opt] = self.opt_param_val[opt_p.id_opt]


    def get_id_opt(self, node_name, id_mod):
        for pos in self.param_pos:
            if pos.node_name == node_name and pos.id_mod == id_mod:
                return pos.id_opt
        return None


    def log_joint(self, params) -> List:
        pass


    def log_likelihood(self, params_opt, node_name):
        params_node = self.get_node_params(params_opt, node_name)
        m_sigma = self.calculate_sigma(params_node)
        if node_name in self.m_cov.keys():
            m_cov = self.m_cov[node_name]
        else:
            m_cov = self.get_matrices(params_node, 'Cov')
        df = sum([p.shape[0] for _, p in self.m_profiles.items()])
        w = wishart.logpdf(m_cov, df=df, scale=m_sigma/df)
        return w


    def log_edge(self, node_name1, node_name2, dist, params_opt):
        # Get positions of parameters which evolve among the phylotree
        pos_evolve = [p for p in self.param_pos if p.node_name == 'Tree']
        prob_edge = 0
        for pos in pos_evolve:
            # print(pos.node_name, pos.mx_type, pos.id_opt, pos.id_mod,node_name1, node_name2)

            if (node_name1 in self.m_cov.keys()) and (pos.mx_type == 'Cov'):
                x = self.get_cov_param(self.m_cov[node_name1], pos.id_mod)
            else:
                id_opt_node1 = self.get_id_opt(node_name1, pos.id_mod)
                if id_opt_node1 is None:
                    raise ValueError('None index returned')
                x = params_opt[id_opt_node1]


            if (node_name2 in self.m_cov.keys()) and (pos.mx_type == 'Cov'):
                m = self.get_cov_param(self.m_cov[node_name2], pos.id_mod)
            else:
                id_opt_node2 = self.get_id_opt(node_name2, pos.id_mod)
                if id_opt_node2 is None:
                    raise ValueError('None index returned')
                m = params_opt[id_opt_node2]

            s = params_opt[pos.id_opt]

            # print('edge_x', x, id_opt_node1, node_name1, pos.id_mod)

            prob_edge += norm.logpdf(x, m, s*dist)
            # print('prob_edge', prob_edge)
            if s < 0:
                self.param_val = params_opt
                raise ValueError('Negative Variance')
        return prob_edge


    def log_post_beta(self, params_opt, node_name):
        prob_beta = self.log_likelihood(params_opt, node_name) + \
                      self.log_prior_beta(params_opt, node_name)
        for node_name2, dist in self.tree[node_name].dist:
            prob_beta += self.log_edge(node_name, node_name2, dist, params_opt)
        return prob_beta


    def log_post_lambda(self, params_opt, node_name):
        prob_lambda = self.log_likelihood(params_opt, node_name) + \
                      self.log_prior_lambda(params_opt, node_name)
        for node_name2, dist in self.tree[node_name].dist:
            prob_lambda += self.log_edge(node_name, node_name2, dist, params_opt)
        return prob_lambda


    def log_post_theta(self, params_opt):
        prob_theta = self.log_prior_theta(params_opt)
        prob_theta += sum(map(lambda x: self.log_likelihood(params_opt, x),
                              self.tree.keys()))
        return prob_theta


    def log_post_psi(self, params_opt, *args):
        prob_psi = self.log_prior_psi(params_opt)
        prob_psi += sum(map(lambda x: self.log_likelihood(params_opt, x),
                            self.tree.keys()))
        return prob_psi


    def log_post_cov(self, params_opt, node_name):
        prob_cov = self.log_likelihood(params_opt, node_name)
        for node_name2, dist in self.tree[node_name].dist:
            # print(node_name, node_name2, dist)
            prob_cov += self.log_edge(node_name, node_name2, dist, params_opt)
        return prob_cov


    def log_post_tree(self, params_opt, *args):
        prob_tree = self.log_prior_tree(params_opt)
        # print('prob_tree', prob_tree)
        for node_name, edges in self.tree.items():
            # TODO
            if len(edges.dist) == 2:  # If root
                continue
            edge = edges.dist[0]
            node_name2, dist = edge
            # print(node_name, node_name2, dist)
            # print('prob_tree', prob_tree)
            prob_tree += self.log_edge(node_name, node_name2, dist, params_opt)
            # print('prob_tree', prob_tree)
        return prob_tree


    def log_post_branch(self, params_opt, *args):
        prob_tree = 0
        # Get the value of the tree-parameter
        param_branch = [params_opt[p.id_opt] for p in \
                        self.param_pos if
                        p.node_name == 'Branch']
        # Change Tree First
        self.tree_init.set_param(param_branch[0])
        self.tree = self.tree_init.nodes

        for node_name, edges in self.tree.items():
            # TODO
            if len(edges.dist) == 2:  # If root
                continue
            edge = edges.dist[0]
            node_name2, dist = edge
            # print(node_name, node_name2, dist)
            # print('prob_tree', prob_tree)
            prob_tree += self.log_edge(node_name, node_name2, dist, params_opt)
            # print('prob_tree', prob_tree)
        return prob_tree


    def get_params(self, tree: SEMBranch, mod_node: SEMModelNode):

        # Common Theta and Psi matrices
        node_names = tree.nodes.keys()
        for i, position in mod_node.param_pos.items():
            mx_type = position[0]
            if mx_type not in {'Psi', 'Theta'}:
                continue
            for name in node_names:
                # Add new params
                self.param_pos += [Parameter(node_name=name,
                                             mx_type=mx_type,
                                             id_opt=self.n_params,
                                             id_mod=i)]
            # Outside of the loop
            # Single parameter for all nodes
            # self.param_val += [mod_node.param_val[i]]
            self.n_params += 1

        # Lambtda and Beta matrices
        node_names = tree.nodes.keys()
        for i, position in mod_node.param_pos.items():
            mx_type = position[0]
            if mx_type not in {'Lambda', 'Beta'}:
                continue
            for name in node_names:
                # Add new paraту
                self.param_pos += [Parameter(node_name=name,
                                             mx_type=mx_type,
                                             id_opt=self.n_params,
                                             id_mod=i)]
                # WITHIN the loop
                # Separate parameters for all nodes
                # self.param_val += [mod_node.param_val[i]]
                self.n_params += 1

        # Covariance Matrices with parameters
        node_names = [name for name, node in tree.nodes.items() if node.type == 'node']
        for i, position in mod_node.param_pos.items():
            mx_type = position[0]
            if mx_type not in {'Cov'}:
                continue
            for name in node_names:
                # Add new params
                self.param_pos += [Parameter(node_name=name,
                                             mx_type=mx_type,
                                             id_opt=self.n_params,
                                             id_mod=i)]
                # WITHIN the loop
                # Separate parameters for all nodes
                # self.param_val += [mod_node.param_val[i]]
                self.n_params += 1

        # Parameters of a tree-process - the same for each node
        # Cov, Beta and Lambda matrices
        for i, position in mod_node.param_pos.items():
            mx_type = position[0]
            if mx_type not in {'Lambda', 'Beta', 'Cov'}:
                continue

            # Add new param
            self.param_pos += [Parameter(node_name='Tree',
                                         mx_type=mx_type,
                                         id_opt=self.n_params,
                                         id_mod=i)]
            # Outside the loop
            # Separate parameters for all nodes
            # self.param_val += [mod_node.param_val[i]]
            self.n_params += 1

        # Add new BRANCH param
        self.param_pos += [Parameter(node_name='Branch',
                                     mx_type='Star',
                                     id_opt=self.n_params,
                                     id_mod=-1)]
        self.n_params += 1

        self.param_val = np.zeros(self.n_params)


    def set_params(self, mod_node: SEMModelNode, dataset: List[SEMData]):
        """

        :param mod_node:
        :param dataset:
        :return:
        """
        # Beta and Lambda Parameters start from zero values

        data_id = 0
        mod_node.load_initial_dataset(dataset[data_id])

        for pos in self.param_pos:
            # Only Psi, Theta and Cov parameters

            if pos.node_name is 'Tree':
                self.param_val[pos.id_opt] = 1
                continue
            if pos.mx_type in {'Beta', 'Lambda'}:
                continue
            if pos.node_name is 'Branch':
                self.param_val[pos.id_opt] = self.b_length/2
                continue
            self.param_val[pos.id_opt] = mod_node.param_val[pos.id_mod]


    def log_prior_psi(self, params_opt, *args):
        """ Inverse Whishart distribution of r0 and rho0"""

        rand_node_name = list(self.tree.keys())[0]
        params_node = self.get_node_params(params_opt, rand_node_name)
        ms_psi = self.get_matrices(params_node, 'Psi')
        prob_psi = invwishart.logpdf(ms_psi, self.p_psi_df, self.p_psi_cov)
        return prob_psi


    def log_prior_beta(self, params_opt, node_name):
        """ Normal """
        params_node = self.get_node_params(params_opt, node_name)
        ms_psi = self.get_matrices(params_node, 'Psi')
        prob_beta = 0
        for pos in self.param_pos:
            if pos.mx_type != 'Beta' or pos.node_name != node_name:
                continue
            prob_beta += norm.logpdf(params_opt[pos.id_opt],
                                     self.p_beta_mean,
                                     self.p_beta_cov * ms_psi[pos.id_mod,
                                                              pos.id_mod])
        return prob_beta


    def log_prior_theta(self, params_opt):
        """ Inverse Gamma distribution of r0 and rho0"""
        # As this parameter is the same through all of the nodes,
        # let take theta from the Theta parameters from the random node
        rand_node_name = list(self.tree.keys())[0]
        params_theta = [params_opt[pos.id_opt] for pos in self.param_pos
                        if pos.mx_type == 'Theta' and pos.node_name == rand_node_name]
        invgamma_theta = partial(invgamma.logpdf,
                                 a=self.p_theta_alpha,
                                 scale=self.p_theta_beta)
        prob_theta = reduce(lambda x, y: x+y,
                            map(invgamma_theta, params_theta))
        return prob_theta


    def log_prior_tree(self, params_opt):
        params_tree = [params_opt[p.id_opt] for p in self.param_pos if p.node_name == 'Tree']
        invgamma_tree = partial(invgamma.logpdf,
                                a=self.p_tree_alpha,
                                scale=self.p_tree_beta)

        prob_tree = reduce(lambda x, y: x+y,
                            map(invgamma_tree, params_tree))
        return prob_tree


    def log_prior_lambda(self, params_opt, node_name):
        """ Normal """
        params_node = self.get_node_params(params_opt, node_name)
        ms_theta = self.get_matrices(params_node)
        prob_lambda = 0
        for pos in self.param_pos:
            if pos.mx_type != 'Lambda' or pos.node_name != node_name:
                continue
            prob_lambda += norm.logpdf(params_opt[pos.id_opt],
                                       self.p_lambda_mean,
                                       self.p_lambda_cov * ms_theta[pos.id_mod,
                                                                    pos.id_mod])
        return prob_lambda


    def log_prior_branch(self, params_opt):
        """ Uniform """
        return 0 # Log(1) = 0


    def prior_params_psi(self):
        """

        :return:
        """
        # for Psi matrix

        # Mean value of all Psi matrices
        m_psi = reduce(lambda x, y: x+y,
                       [self.get_matrices(params, 'Psi')
                        for _, params in self.param_leaf.items()])
        m_psi = m_psi / len(self.param_leaf)

        # Dimension of psi matrix
        psi_dim = m_psi.shape[0]
        # Total number of all samples is a degree of freedom
        p_psi_df = sum([p.shape[0] for _, p in self.m_profiles.items()])
        p_psi_cov = m_psi * (p_psi_df - psi_dim - 1)
        return p_psi_df, p_psi_cov


    def prior_params_coefs(self, mx_type):
        """
        Parameters of the prior distribution of params if Beta of Lambda
        :return: mean and variance of Parameters
        """
        coef_init = []
        for pos in self.param_pos:
            for _, param in self.param_leaf.items():
                if pos.node_name != 'Tree' and pos.mx_type == mx_type:
                    coef_init += [param[pos.id_mod]]

        if not coef_init:
            coef_init = 0
        coef_mean = np.mean(coef_init)
        coef_var = np.var(coef_init) + 1
        return coef_mean, coef_var


    def prior_params_theta(self):
        """

        :return:
        """

        theta_init = []
        for pos in self.param_pos:
            for _, param in self.param_leaf.items():
                if pos.node_name != 'Tree' and pos.mx_type == 'Theta':
                    theta_init += [param[pos.id_mod]]

        if not theta_init:
            theta_init = 0

        # Total number of all samples is a degree of freedom
        df = sum([p.shape[0] for _, p in self.m_profiles.items()])
        p_theta_alpha = df/2
        p_theta_beta = np.median(theta_init) * (p_theta_alpha - 1)
        return p_theta_alpha, p_theta_beta


    def prior_params_tree(self):
        """
        Parameters of a prior distribution of phylogenetic Wiener process
        Parameter Sigma of a Wiener process is prior distributed by inv-gamma
        :return:
        """
        id = {name:i for i, name in enumerate(list(self.tree.keys()))}
        n_nodes = len(id)
        dist_mx = np.zeros((n_nodes, n_nodes))

        for node1, edges in self.tree.items():
            for node2, dist in edges.dist:
                dist_mx[id[node1], id[node2]] = dist
                dist_mx[id[node2], id[node1]] = dist

        # while np.count_nonzero(dist_mx) < (n_nodes ** 2 - n_nodes):
        for _ in range(20):
            for i, j in combinations(range(n_nodes), 2):
                if dist_mx[i,j] > 0:
                    continue
                row_i = dist_mx[i]
                row_j = dist_mx[j]
                value = (row_i + row_j) * (row_i > 0) * (row_j > 0)
                dist_mx[i, j] = dist_mx[j, i] = - max(np.unique(value))
            dist_mx = np.abs(dist_mx)

        evolve_rate = []
        for node1, node2 in combinations(self.m_cov.keys(), 2):
            mx_cov_dist = np.abs(self.m_cov[node1] - self.m_cov[node2])
            elements = mx_cov_dist[np.triu_indices(len(mx_cov_dist))]
            norm_elements = elements / dist_mx[id[node2], id[node1]]
            evolve_rate += list(norm_elements)



        df = np.mean([p.shape[0] for _, p in self.m_profiles.items()])
        p_theta_alpha = df/2
        # p_theta_alpha = 4
        p_theta_beta = np.percentile(evolve_rate, 75) * (p_theta_alpha - 1)
        # print(p_theta_alpha, p_theta_beta)
        return p_theta_alpha, p_theta_beta


    def constraint_theta(self, params_opt, node_name):
        params_node = self.get_node_params(params_opt, node_name)
        ms_theta = self.get_matrices(params_node, 'Theta')
        return sum(ms_theta.diagonal() >= 0) - ms_theta.shape[0]


    def constraint_psi(self, params_opt, node_name):
        params_node = self.get_node_params(params_opt, node_name)
        ms_psi = self.get_matrices(params_node, 'Psi')
        # return np.linalg.det(ms['Psi']) - 1e-6
        return sum(np.linalg.eig(ms_psi)[0] > 0) - ms_psi.shape[0]


    def constraint_sigma(self, params_opt, node_name):
        params_node = self.get_node_params(params_opt, node_name)
        m_sigma = self.calculate_sigma(params_node)
        # return np.linalg.det(m_sigma) - 1e-6
        return sum(np.linalg.eig(m_sigma)[0] > 0) - m_sigma.shape[0]


    def constraint_cov(self, params_opt, node_name):
        params_node = self.get_node_params(params_opt, node_name)
        m_cov = self.get_matrices(params_node, 'Cov')
        # return np.linalg.det(m_sigma) - 1e-6
        return sum(np.linalg.eig(m_cov)[0] > 0) - m_cov.shape[0]


    def constraint_tree(self, params_opt, node_name):
        params_tree = [params_opt[p.id_opt] > 0 for p in self.param_pos
                       if p.node_name == node_name]
        return sum(params_tree) - len(params_tree)


    def calculate_sigma(self, params):
        """
        Sigma matrix calculated from the model
        """
        ms = self.get_matrices(params)
        m_beta = ms['Beta']
        m_lambda = ms['Lambda']
        m_psi = ms['Psi']
        m_theta = ms['Theta']

        m_c = np.linalg.pinv(np.identity(m_beta.shape[0]) - m_beta)
        return m_lambda @ m_c @ m_psi @ m_c.T @ m_lambda.T + m_theta


    def constraint_branch(self, params_opt, *args):
        param_branch = [params_opt[p.id_opt] for p in \
                        self.param_pos if
                        p.node_name == 'Branch']
        if (0 <= param_branch[0] <= self.b_length):
            return 0
        else:
            return -1



