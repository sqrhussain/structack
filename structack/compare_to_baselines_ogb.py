import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.global_attack import DICE, Random, Metattack
from structack.structack import StructackBothEndsGreedy
import pandas as pd
import time
import os
from ogb.nodeproppred import NodePropPredDataset
from structack.compare_to_baselines import *
from scipy.sparse import csr_matrix


def apply_perturbation(model_builder, attack, data, ptb_rate, cuda, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


    device = torch.device("cuda" if cuda else "cpu")

    adj, features, labels, idx_train, idx_val, idx_test = ogb_to_deeprobust(data)
    idx_unlabeled = np.union1d(idx_val, idx_test)


    n_perturbations = int(ptb_rate * (adj.sum()//2))

    if model_builder == build_mettack:
        adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
        features = torch.FloatTensor(features).to(device)
        labels = torch.LongTensor(labels).to(device)

    # build the model
    model = model_builder(adj, features, labels, idx_train, device)
    
    tick = time.time()
    # perform the attack
    modified_adj = attack(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled)
    elapsed = time.time() - tick

    modified_adj = modified_adj.to(device)
    return modified_adj, elapsed


def ogb_to_deeprobust(dataset):
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph, label = dataset[0]
    label = label.flatten()
    edge_index = graph['edge_index']
    # reversed_edges = np.array([edge_index[1],edge_index[0]])
    # edge_index = np.concatenate((edge_index,reversed_edges),axis=1)

    adj = csr_matrix((np.ones([edge_index.shape[1]]), edge_index))
    adj.data = np.clip(adj.data,0,1)
    features = graph['node_feat']
    return adj,features,label,train_idx,valid_idx,test_idx
    

def pre_test_data_ogb(data,device):
    _, features, labels, idx_train, idx_val, idx_test = ogb_to_deeprobust(data)
    features = torch.FloatTensor(features).to(device)
    labels = torch.LongTensor(labels).to(device)
    return features, labels, idx_train, idx_val, idx_test

def main_ogb():
    df_path = 'reports/initial_eval-ogb-mag.csv'
    datasets = 'ogbn-mag'.split()
    for dataset in datasets:
        for attack, model_builder, model_name in zip(attacks,model_builders, model_names):
            data = NodePropPredDataset(name = dataset)
            device = torch.device("cuda" if cuda else "cpu")
            print(device)
            adj = ogb_to_deeprobust(data)[0]
            print('converted data')
            adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
            print('moved adj')
            print(test(adj,data, cuda, pre_test_data_ogb))
            df = pd.DataFrame()
            for perturbation_rate in [0.005,0.010,0.015,0.020,0.050]:
                for seed in range(1):
                    modified_adj, elapsed = apply_perturbation(model_builder, attack, data, perturbation_rate, cuda, seed)
                    acc = test(modified_adj, data, cuda, pre_test_data_ogb,nhid=16)
                    row = {'dataset':dataset, 'attack':model_name, 'seed':seed, 'acc':acc, 'perturbation_rate':perturbation_rate,'elapsed':elapsed}
                    print(row)
                    df = df.append(row, ignore_index=True)
            cdf = pd.DataFrame(columns=df.columns)
            if os.path.exists(df_path):
                cdf = pd.read_csv(df_path)
            df = pd.concat([cdf,df])
            df.to_csv(df_path,index=False)



if __name__ == '__main__':
    main_ogb()