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
    def __init__(self, degree_percentile_range=[0,1], device='cpu'):
        super(Structack, self).__init__(None, None, attack_structure=True, attack_features=False, device=device)
        
        self.modified_adj = None
        self.frm, self.to = degree_percentile_range

    def get_nodes_with_degree_percentile(self, G, frm, to):

        nodes = sorted(G.degree, key=lambda x: x[1])
        length = len(nodes)
        frm = int(length*frm)
        to =  int(length*to +1)
        nodes = nodes[frm:to]
        return [x[0] for x in nodes]

    def get_purturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(utils.to_scipy(adj), create_using=nx.Graph)
        
        # perturb
        node_pool = self.get_nodes_with_degree_percentile(graph,self.frm,self.to)
        rows = np.random.choice(node_pool,n_perturbations)
        cols = np.random.choice(graph.nodes(),n_perturbations)
        edges = [[u,v] for u,v in zip(rows, cols)]
        graph.add_edges_from(edges)

        modified_adj, _ = utils.to_tensor(nx.to_scipy_sparse_matrix(graph), np.array([[0]]), None, self.device)
        return modified_adj
    

    def attack(self, ori_adj, n_perturbations):
        self.modified_adj = self.get_purturbed_adj(ori_adj,n_perturbations)

class StructackOneEnd(Structack):
    def get_purturbed_adj(self, adj):
        pass

class StructackBothEnds(Structack):
    def get_purturbed_adj(self, adj):
        pass

