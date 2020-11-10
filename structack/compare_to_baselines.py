
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.global_attack import DICE, Random, Metattack, PGDAttack, MinMax
from structack.structack import StructackBase
from structack.structack import StructackDegreeRandomLinking, StructackDegree, StructackDegreeDistance,StructackDistance
from structack.structack import StructackEigenvectorCentrality, StructackBetweennessCentrality, StructackClosenessCentrality
from structack.structack import StructackPageRank, StructackKatzSimilarity, StructackCommunity
import structack.node_selection as ns
import structack.node_connection as nc
# from structack.calc_unnoticeability import *
import pandas as pd
import time
import os


def postprocess_adj(adj):
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def attack_dice(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, labels, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)

def attack_random(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)

def attack_structack1(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)

def attack_structack2(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)

def attack_structack2_greedy(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)

def attack_structack_fold(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)

def attack_structack_distance(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)

def attack_structack_only_distance(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)


def attack_structack(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)


def attack_mettaack(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=False)
    return model.modified_adj


def attack_pgd(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(features, adj, labels, idx_train, n_perturbations)
    return model.modified_adj


def attack_minmax(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(features, adj, labels, idx_train, n_perturbations)
    return model.modified_adj


def attack_structack_eigenvector_centrality(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)

def attack_structack_betwenness_centrality(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)

def attack_structack_closeness_centrality(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)

def attack_structack_pagerank(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)

def attack_structack_katz_similarity(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)

def attack_structack_community(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)

def build_random(adj=None, features=None, labels=None, idx_train=None, device=None):
    return Random()

def build_dice(adj=None, features=None, labels=None, idx_train=None, device=None):
    return DICE()

# def build_structack1(adj=None, features=None, labels=None, idx_train=None, device=None):
#     return StructackOneEnd(degree_percentile_range=[0,.1])
#
# def build_structack2(adj=None, features=None, labels=None, idx_train=None, device=None):
#     return StructackBothEnds(degree_percentile_range=[0,.1,0,.1])

def build_structack2_greedy(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackDegreeRandomLinking()

def build_structack_fold(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackDegree()

def build_structack_distance(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackDegreeDistance()

def build_structack_only_distance(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackDistance()

def build_structack_eigenvector_centrality(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackEigenvectorCentrality()

def build_structack_betweenness_centrality(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackBetweennessCentrality()

def build_structack_closeness_centrality(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackClosenessCentrality()

def build_structack_pagerank(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackPageRank()

def build_structack_katz_similarity(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackKatzSimilarity()

def build_structack_community(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackCommunity()

def build_mettack(adj=None, features=None, labels=None, idx_train=None, device=None):    
    lambda_ = 0
    
    # Setup Surrogate Model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
            dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)
    
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
            attack_structure=True, attack_features=False, device=device, lambda_=lambda_)
    model = model.to(device)
    return model


def build_pgd(adj=None, features=None, labels=None, idx_train=None, device=None):
    # Setup Victim Model
    victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
            dropout=0.5, weight_decay=5e-4, device=device)

    victim_model = victim_model.to(device)
    victim_model.fit(features, adj, labels, idx_train)
    return PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device)

def build_minmax(adj=None, features=None, labels=None, idx_train=None, device=None):
    # Setup Victim Model
    victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
            dropout=0.5, weight_decay=5e-4, device=device)

    victim_model = victim_model.to(device)
    victim_model.fit(features, adj, labels, idx_train)
    return MinMax(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device)

def build_custom(node_selection, node_connection):

    class StructackTemp(StructackBase):

        def node_selection(self, graph, n):
            return node_selection(graph, n)

        def node_connection(self, adj, nodes, n_perturbations):
            return node_connection(adj, nodes, n_perturbations)

    return StructackTemp()


def apply_structack(model, attack, data, ptb_rate, cuda, seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")

    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)


    n_perturbations = int(ptb_rate * (adj.sum()//2))
    print(f'n_perturbations = {n_perturbations}')

        
    tick = time.time()
    # perform the attack
    modified_adj = attack(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled)
    elapsed = time.time() - tick
    modified_adj = modified_adj.to(device)
    return modified_adj, elapsed

def apply_perturbation(model_builder, attack, data, ptb_rate, cuda, seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")

    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)


    n_perturbations = int(ptb_rate * (adj.sum()//2))
    print(f'n_perturbations = {n_perturbations}')

    if model_builder == build_mettack:
        adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

    # build the model
    model = model_builder(adj, features, labels, idx_train, device)
        
    tick = time.time()
    # perform the attack
    modified_adj = attack(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled)
    elapsed = time.time() - tick
    modified_adj = modified_adj.to(device)
    return modified_adj, elapsed

def pre_test_data(data,device):
    features, labels = data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    _ , features, labels = preprocess(data.adj, features, labels, preprocess_adj=False, sparse=True, device=device)
    return features, labels, idx_train, idx_val, idx_test

def test(adj, data, cuda, data_prep,nhid=16):
    ''' test on GCN '''
    device = torch.device("cuda" if cuda else "cpu")
    features, labels, idx_train, idx_val, idx_test = data_prep(data,device)

    gcn = GCN(nfeat=features.shape[1],
              nhid=nhid,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)

    optimizer = optim.Adam(gcn.parameters(),
                           lr=0.01, weight_decay=5e-4)

    gcn.fit(features, adj, labels, idx_train) # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    return acc_test.item()



def main():
    df_path = 'reports/eval/init_eval_garbage.csv'
    # datasets = ['citeseer', 'cora', 'cora_ml', 'polblogs', 'pubmed']
    datasets = ['cora']
    for dataset in datasets:
        for attack, model_builder, model_name in zip(attacks,model_builders, model_names):
            print('attack ' + model_name)
            data = Dataset(root='/tmp/', name=dataset)
            # adj,_,_ = preprocess(data.adj, data.features, data.labels, preprocess_adj=False, sparse=True, device=torch.device("cuda" if cuda else "cpu"))
            # acc = test(adj, data, cuda, pre_test_data)
            # row = {'dataset':dataset, 'attack':'Clean', 'seed':None, 'acc':acc}
            # print(row)
            # df = df.append(row, ignore_index=True)
            for perturbation_rate in [0.05]:#,0.10,0.15,0.20]:
                for seed in range(3):
                    modified_adj, elapsed = apply_perturbation(model_builder, attack, data, perturbation_rate, cuda and (dataset!='pubmed'), seed)
                    # print((to_scipy(modified_adj) != to_scipy(modified_adj1)).nnz==0)
                    acc = test(modified_adj, data, cuda, pre_test_data)
                    row = {'dataset':dataset, 'attack':model_name, 'seed':seed, 'acc':acc, 'perturbation_rate':perturbation_rate,'elapsed':elapsed}
                    print(row)
                    cdf = pd.DataFrame()
                    if os.path.exists(df_path):
                        cdf = pd.read_csv(df_path)
                    cdf = cdf.append(row, ignore_index=True)
                    cdf.to_csv(df_path,index=False)


def combination():
    
    df_path = 'reports/eval/initial_comb_eval.csv'

    selection_options = [
                [ns.get_nodes_with_lowest_eigenvector_centrality,'eigenvector'],
                [ns.get_nodes_with_lowest_betweenness_centrality,'betweenness'],
                [ns.get_nodes_with_lowest_closeness_centrality,'closeness'],
                [ns.get_nodes_with_lowest_pagerank,'pagerank'],
                [ns.get_nodes_with_lowest_degree,'degree'],
                [ns.get_random_nodes,'random'],
            ]

    connection_options = [
                [nc.distance_hungarian_connection,'distance'],
                [nc.katz_connection,'katz'],
                [nc.community_connection,'community'],
                [nc.random_connection,'random'],
            ]

    datasets = ['citeseer', 'cora', 'cora_ml', 'polblogs', 'pubmed']
    # datasets = ['cora']
    for dataset in datasets:
        data = Dataset(root='/tmp/', name=dataset)
        for selection, selection_name in selection_options:
            for connection, connection_name in connection_options:
                print(f'attack [{selection_name}]*[{connection_name}]')
                for perturbation_rate in [0.05]:#,0.10,0.15,0.20]:
                    modified_adj, elapsed = apply_structack(build_custom(selection, connection), attack_structack, data, perturbation_rate, cuda and (dataset!='pubmed'), seed=0)
                    for seed in range(10):
                        acc = test(modified_adj, data, cuda, pre_test_data)
                        row = {'dataset':dataset, 'selection':selection_name, 'connection':connection_name,
                                'seed':seed, 'acc':acc, 'perturbation_rate':perturbation_rate,'elapsed':elapsed}
                        print(row)
                        cdf = pd.DataFrame()
                        if os.path.exists(df_path):
                            cdf = pd.read_csv(df_path)
                        cdf = cdf.append(row, ignore_index=True)
                        cdf.to_csv(df_path,index=False)




# The following lists should be correspondent


# attacks = [
#     [attack_random, 'Random', build_random],
#     [attack_dice, 'DICE', build_dice],
# ]

attacks = [
    # attack_random,
    # attack_dice,
    # attack_structack_fold, 
    # attack_structack_only_distance,
    # attack_structack_distance,
    # attack_mettaack,
    attack_structack,
    attack_structack, 
    attack_structack, 
    attack_structack,
    attack_structack,
    attack_structack,
]
model_names = [
    # 'Random',
    # 'DICE',
    # 'StructackGreedyFold', # this is StructackDegree in the paper
    # 'StructackOnlyDistance', # this is StructackDistance in the paper
    # 'StructackDistance', # this is Structack in the paper
    # 'Metattack',
    'StructackEigenvectorCentrality',
    'StructackBetweennessCentrality',
    'StructackClosenessCentrality',
    'StructackPageRank',
    'StructackKatzSimilarity',
    'StructackCommunity',
]
model_builders = [
    # build_random,
    # build_dice,
    # build_structack_fold,
    # build_structack_only_distance,
    # build_structack_distance,
    # build_mettack,
    build_structack_eigenvector_centrality, 
    build_structack_betweenness_centrality, 
    build_structack_closeness_centrality, 
    build_structack_pagerank,
    build_structack_katz_similarity,
    build_structack_community
]
cuda = torch.cuda.is_available()

if __name__ == '__main__':
    combination()