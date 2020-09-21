
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.global_attack import DICE, Random, Metattack
from structack.structack import StructackOneEnd, StructackBothEnds, StructackBothEndsGreedy
import pandas as pd
import time



def attack_dice(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, labels, n_perturbations)

def attack_random(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)

def attack_structack1(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)

def attack_structack2(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)

def attack_structack2_greedy(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(adj, n_perturbations)


def attack_mettaack(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
    return model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=False)

def build_random(adj=None, features=None, labels=None, idx_train=None, device=None):
    return Random()

def build_dice(adj=None, features=None, labels=None, idx_train=None, device=None):
    return DICE(),

def build_structack1(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackOneEnd(degree_percentile_range=[0,.1])

def build_structack2(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackBothEnds(degree_percentile_range=[0,.1,0,.1])

def build_structack2_greedy(adj=None, features=None, labels=None, idx_train=None, device=None):
    StructackBothEndsGreedy()

def build_mettack(adj=None, features=None, labels=None, idx_train=None, device=None):

    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, device=device)
    
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


    device = torch.device("cuda:0" if cuda else "cpu")

    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)

    n_perturbations = int(ptb_rate * (adj.sum()//2))

    # build the model
    model = model_builder(adj, features, labels, idx_train, device)
    
    tick = time.time()
    # perform the attack
    attack(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled)
    elapsed = time.time() - tick

    modified_adj = model.modified_adj

    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=True, device=device)

    modified_adj = normalize_adj(modified_adj)
    modified_adj = sparse_mx_to_torch_sparse_tensor(modified_adj)
    modified_adj = modified_adj.to(device)
    return modified_adj, elapsed


def test(adj,data, cuda):
    ''' test on GCN '''
    features, labels = data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)

    device = torch.device("cuda:0" if cuda else "cpu")

    _ , features, labels = preprocess(data.adj, features, labels, preprocess_adj=False, sparse=True, device=device)

    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
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
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test.item()),
    #       "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()




def main():
    df_path = 'reports/initial_eval.csv'
    attacks = [
        # attack_random,
        # attack_dice,
        # attack_structack1,
        # attack_structack2,
        # attack_structack2_greedy,
        attack_mettaack,
    ]
    model_names = [
        'Random',
        'DICE',
        'StructackOneEnd',
        'StructackBothEnds',
        'StructackBothEndsGreedy',
        'Metattack',
    ]
    model_builders = [
        # build_random,
        # build_dice,
        # build_structack1,
        # build_structack2,
        # build_structack2_greedy,
        build_mettack,
    ]
    datasets = ['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed']
    cuda = torch.cuda.is_available()
    for dataset in datasets:
        for attack, model_builder, model_name in zip(attacks,model_builders, model_names):
            data = Dataset(root='/tmp/', name=dataset)
            adj,_,_ = preprocess(data.adj, data.features, data.labels, preprocess_adj=False, sparse=True, device=torch.device("cuda:0" if cuda else "cpu"))
            acc = test(adj, data, cuda)
            row = {'dataset':dataset, 'attack':'Clean', 'seed':None, 'acc':acc}
            print(row)
            df = pd.DataFrame()
            df = df.append(row, ignore_index=True)
            for perturbation_rate in [0.05,0.01,0.10,0.15,0.20]:
                for seed in range(10):
                    modified_adj, elapsed = apply_perturbation(model_builder, attack, data, perturbation_rate, cuda, seed)
                    acc = test(modified_adj, data, cuda)
                    row = {'dataset':dataset, 'attack':model_name, 'seed':seed, 'acc':acc, 'perturbation_rate':perturbation_rate,'elapsed':elapsed}
                    print(row)
                    df = df.append(row, ignore_index=True)
            cdf = pd.read_csv(df_path)
            df = pd.concat([cdf,df])
            df.to_csv(df_path,index=False)

if __name__ == '__main__':
    main()