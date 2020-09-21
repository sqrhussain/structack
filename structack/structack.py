import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from deeprobust.graph import utils
from deeprobust.graph.global_attack import BaseAttack
import networkx as nx

class Structack(BaseAttack):
    def __init__(self):
        super(Structack, self).__init__(None, None, attack_structure=True, attack_features=False, device='cpu')
        self.modified_adj = None

    def get_nodes_with_degree_percentile(self, G, frm, to):
        nodes = sorted(G.degree, key=lambda x: x[1])
        length = len(nodes)
        frm = int(length*frm)
        to =  int(length*to +1)
        nodes = nodes[frm:to]
        return [x[0] for x in nodes]

    def get_nodes_with_lowest_degree(self, G, n):
        nodes = sorted(G.degree, key=lambda x: x[1])
        nodes = nodes[:n]
        return [x[0] for x in nodes]

    def get_purturbed_adj(self, adj, n_perturbations):
        pass

    def attack(self, ori_adj, n_perturbations):
        self.modified_adj = self.get_purturbed_adj(ori_adj,n_perturbations)

class StructackOneEnd(Structack):
    def __init__(self, degree_percentile_range=[0,1]):
        super(StructackOneEnd, self).__init__()
        self.frm, self.to = degree_percentile_range
    def get_purturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
        
        # perturb
        node_pool = self.get_nodes_with_degree_percentile(graph,self.frm,self.to)
        rows = np.random.choice(node_pool,n_perturbations)
        cols = np.random.choice(graph.nodes(),n_perturbations)
        edges = [[u,v] for u,v in zip(rows, cols)]
        graph.add_edges_from(edges)

        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj

class StructackBothEnds(Structack):
    def __init__(self, degree_percentile_range=[0,1,0,1]):
        super(StructackBothEnds, self).__init__()
        self.frm1, self.to1, self.frm2, self.to2 = degree_percentile_range
    def get_purturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
        
        # perturb
        rows = np.random.choice(self.get_nodes_with_degree_percentile(graph,self.frm1,self.to1),n_perturbations)
        cols = np.random.choice(self.get_nodes_with_degree_percentile(graph,self.frm2,self.to2),n_perturbations)
        edges = [[u,v] for u,v in zip(rows, cols)]
        graph.add_edges_from(edges)

        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj

class StructackBothEndsGreedy(Structack):
    def __init__(self):
        super(StructackBothEndsGreedy, self).__init__()
        self.modified_adj = None

    def get_purturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
        n = adj.shape[0]

        # select nodes
        nodes = self.get_nodes_with_lowest_degree(graph,2*n_perturbations)
        np.random.shuffle(nodes)

        # perturb
        rows = nodes[:n_perturbations]
        cols = nodes[n_perturbations:]
        edges = [[u,v] for u,v in zip(rows, cols)]
        graph.add_edges_from(edges)

        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj