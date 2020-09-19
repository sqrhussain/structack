
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.global_attack import DICE, Random
from structack.structack import StructackOneEnd, StructackBothEnds
import pandas as pd



def attack_dice(model, adj, labels, n_perturbations):
    model.attack(adj, labels, n_perturbations)

def attack_random(model, adj, labels, n_perturbations):
    model.attack(adj, n_perturbations)

def attack_structack1(model, adj, labels, n_perturbations):
    model.attack(adj, n_perturbations)

def attack_structack2(model, adj, labels, n_perturbations):
    model.attack(adj, n_perturbations)


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
    attacks = [attack_random,attack_dice,
    attack_structack1,attack_structack2]
    models = [Random(), DICE(),
     StructackOneEnd(degree_percentile_range=[0,.1]), StructackBothEnds(degree_percentile_range=[0,.1,0,.1])]
    datasets = ['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed']
    perturbation_rate = 0.05
    cuda = torch.cuda.is_available()
    df = pd.DataFrame()
    for dataset in datasets:
        for attack, model in zip(attacks,models):
            data = Dataset(root='/tmp/', name=dataset)
            adj,_,_ = preprocess(data.adj, data.features, data.labels, preprocess_adj=False, sparse=True, device=torch.device("cuda:0" if cuda else "cpu"))
            acc = test(adj, data, cuda)
            row = {'dataset':dataset, 'attack':'Clean', 'seed':None, 'acc':acc}
            print(row)
            df = df.append(row, ignore_index=True)
            for seed in range(10):
                modified_adj = apply_perturbation(model, attack, data, perturbation_rate, cuda, seed)
                acc = test(modified_adj, data, cuda)
                row = {'dataset':dataset, 'attack':model.__class__.__name__, 'seed':seed, 'acc':acc}
                print(row)
                df = df.append(row, ignore_index=True)
    df.to_csv('reports/initial_eval.csv',index=False)

if __name__ == '__main__':
    main()