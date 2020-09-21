
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.global_attack import DICE, Random
from structack.structack import StructackOneEnd, StructackBothEnds, StructackBothEndsGreedy
import pandas as pd



def attack_dice(model, adj, labels, n_perturbations):
    model.attack(adj, labels, n_perturbations)

def attack_random(model, adj, labels, n_perturbations):
    model.attack(adj, n_perturbations)

def attack_structack1(model, adj, labels, n_perturbations):
    model.attack(adj, n_perturbations)

def attack_structack2(model, adj, labels, n_perturbations):
    model.attack(adj, n_perturbations)

def attack_structack2_greedy(model, adj, labels, n_perturbations):
    model.attack(adj, n_perturbations)


def arrack_mettaack(model, adj, labels, n_perturbations, features, idx_train, idx_unlabeled):
    pass

def build_random(surrogate=None, victim_model=None, adj=None, features=None, device=None, lambda_=None):
    return Random()

def build_dice(surrogate=None, victim_model=None, adj=None, features=None, device=None, lambda_=None):
    return DICE(),

def build_dice(surrogate=None, victim_model=None, adj=None, features=None, device=None, lambda_=None):
    return StructackOneEnd(degree_percentile_range=[0,.1])

def build_dice(surrogate=None, victim_model=None, adj=None, features=None, device=None, lambda_=None):
    return StructackBothEnds(degree_percentile_range=[0,.1,0,.1])

def build_mettack(surrogate=None, victim_model=None, adj=None, features=None, device=None, lambda_=None):
    return Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
            attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

def build_pgd(surrogate=None, victim_model=None, adj=None, features=None, device=None, lambda_=None):
    return PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device)

def build_minmax(surrogate=None, victim_model=None, adj=None, features=None, device=None, lambda_=None):
    return MinMax(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device)


def apply_perturbation(model, attack, data, ptb_rate, cuda, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


    device = torch.device("cuda:0" if cuda else "cpu")

    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    # idx_unlabeled = np.union1d(idx_val, idx_test)

    n_perturbations = int(ptb_rate * (adj.sum()//2))

    # This depends on the model, how would you manage it?
    attack(model, adj, labels, n_perturbations)
    modified_adj = model.modified_adj

    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=True, device=device)

    modified_adj = normalize_adj(modified_adj)
    modified_adj = sparse_mx_to_torch_sparse_tensor(modified_adj)
    modified_adj = modified_adj.to(device)
    return modified_adj


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
               attack_random,
               attack_dice,
               attack_structack1,
               attack_structack2,
               attack_structack2_greedy,
               ]
    models = [
              Random(),
              DICE(),
              StructackOneEnd(degree_percentile_range=[0,.1]),
              StructackBothEnds(degree_percentile_range=[0,.1,0,.1]),
              StructackBothEndsGreedy(),
              ]
    datasets = ['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed']
    cuda = torch.cuda.is_available()
    for dataset in datasets:
        for attack, model in zip(attacks,models):
            data = Dataset(root='/tmp/', name=dataset)
            adj,_,_ = preprocess(data.adj, data.features, data.labels, preprocess_adj=False, sparse=True, device=torch.device("cuda:0" if cuda else "cpu"))
            acc = test(adj, data, cuda)
            row = {'dataset':dataset, 'attack':'Clean', 'seed':None, 'acc':acc}
            print(row)
            df = pd.DataFrame()
            df = df.append(row, ignore_index=True)
            for perturbation_rate in [0.01]:#0.05,0.10,0.15,0.20]:
                for seed in range(10):
                    modified_adj = apply_perturbation(model, attack, data, perturbation_rate, cuda, seed)
                    acc = test(modified_adj, data, cuda)
                    row = {'dataset':dataset, 'attack':model.__class__.__name__, 'seed':seed, 'acc':acc, 'perturbation_rate':perturbation_rate}
                    print(row)
                    df = df.append(row, ignore_index=True)
            cdf = pd.read_csv(df_path)
            df = pd.concat([cdf,df])
            df.to_csv(df_path,index=False)

if __name__ == '__main__':
    main()