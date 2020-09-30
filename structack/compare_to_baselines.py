
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.global_attack import DICE, Random, Metattack
from structack.structack import StructackGreedyRandom, StructackGreedyFold, StructackDistance
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


def attack_mettaack(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=False)
    return model.modified_adj


def build_random(adj=None, features=None, labels=None, idx_train=None, device=None):
    return Random()

def build_dice(adj=None, features=None, labels=None, idx_train=None, device=None):
    return DICE()

def build_structack1(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackOneEnd(degree_percentile_range=[0,.1])

def build_structack2(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackBothEnds(degree_percentile_range=[0,.1,0,.1])

def build_structack2_greedy(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackGreedyRandom()

def build_structack_fold(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackGreedyFold()

def build_structack_distance(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackDistance()

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
    return PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device)

def build_minmax(adj=None, features=None, labels=None, idx_train=None, device=None):
    return MinMax(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device)


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
    df_path = 'reports/eval/initial_eval-citeseer.csv'
    datasets = ['citeseer', 'cora', 'cora_ml', 'polblogs', 'pubmed']
    # datasets = ['citeseer']
    for dataset in datasets:
        for attack, model_builder, model_name in zip(attacks,model_builders, model_names):
            data = Dataset(root='/tmp/', name=dataset)
            # adj,_,_ = preprocess(data.adj, data.features, data.labels, preprocess_adj=False, sparse=True, device=torch.device("cuda" if cuda else "cpu"))
            # acc = test(adj, data, cuda, pre_test_data)
            # row = {'dataset':dataset, 'attack':'Clean', 'seed':None, 'acc':acc}
            # print(row)
            # df = df.append(row, ignore_index=True)
            for perturbation_rate in [0.05]: #,0.01,0.10,0.15,0.20]:
                for seed in range(10):
                    modified_adj, elapsed = apply_perturbation(model_builder, attack, data, perturbation_rate, cuda, seed)
                    acc = test(modified_adj, data, cuda, pre_test_data)
                    row = {'dataset':dataset, 'attack':model_name, 'seed':seed, 'acc':acc, 'perturbation_rate':perturbation_rate,'elapsed':elapsed}
                    print(row)
                    cdf = pd.DataFrame()
                    if os.path.exists(df_path):
                        cdf = pd.read_csv(df_path)
                    cdf = cdf.append(row, ignore_index=True)
                    cdf.to_csv(df_path,index=False)


attacks = [
    # attack_random,
    # attack_dice,
    # attack_structack2_greedy,
    # attack_structack1,
    # attack_structack2,
    attack_structack_fold,
    # attack_structack_distance,
    # attack_mettaack,
]
model_names = [
    # 'Random',
    # 'DICE',
    # 'StructackGreedyRandom',
    # 'StructackOneEnd',
    # 'StructackBothEnds',
    'StructackGreedyFold',
    # 'StructackDistanceMod',
    # 'Metattack',
]
model_builders = [
    # build_random,
    # build_dice,
    # build_structack2_greedy,
    # build_structack1,
    # build_structack2,
    build_structack_fold,
    # build_structack_distance,
    # build_mettack,
]
cuda = torch.cuda.is_available()

if __name__ == '__main__':
    main()