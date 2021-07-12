from structack.compare_to_baselines import build_mettack, attack_mettaack
from deeprobust.graph.global_attack import DICE
import networkx as nx
import torch
import numpy as np
from deeprobust.graph.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def attack_dice(model, adj, labels, n_perturbations):
    model.attack(adj, labels, n_perturbations)
    modified_adj = model.modified_adj
    return modified_adj


fh = open("../graph-directions/data/graphs/processed/structacka/structacka.cites", "rb")
G = nx.read_edgelist(fh)
G = nx.relabel_nodes(G, lambda x:int(x))
nodes = list(G.nodes)
print(nodes)
print(G.edges)
adj = nx.to_scipy_sparse_matrix(G)

labels = np.array([0,0,0,0,0,1,1,1,1,1])
features = np.eye(10)
idx_train = np.array([3,8])
idx_unlabeled = np.array([0,1,2,4,5,6,7,9])
n_perturbations = 2

# model = DICE()
model = build_mettack(adj, features, labels, idx_train, device)
# modified_adj = nx.from_scipy_sparse_matrix(attack_dice(model, adj, labels , n_perturbations))
modified_adj = nx.from_scipy_sparse_matrix(to_scipy(attack_mettaack(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled)))


modified_adj = nx.relabel_nodes(modified_adj, lambda x:nodes[x])
print(modified_adj.edges)