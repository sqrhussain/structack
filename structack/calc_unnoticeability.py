
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.global_attack import DICE, Random, Metattack
from structack.structack import StructackDegreeRandomLinking, StructackDegree, StructackDegreeDistance,StructackDistance
import pandas as pd
import time
import os
from scipy.stats import wilcoxon
import networkx as nx


def postprocess_adj(adj):
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


def attack_mettaack(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=False)
    return to_scipy(model.modified_adj)


def build_random(adj=None, features=None, labels=None, idx_train=None, device=None):
    return Random()

def build_dice(adj=None, features=None, labels=None, idx_train=None, device=None):
    return DICE()

def build_structack1(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackOneEnd(degree_percentile_range=[0,.1])

def build_structack2(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackBothEnds(degree_percentile_range=[0,.1,0,.1])

def build_structack2_greedy(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackDegreeRandomLinking()

def build_structack_fold(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackDegree()

def build_structack_distance(adj=None, features=None, labels=None, idx_train=None, device=None):
    return StructackDegreeDistance()

def build_structack_only_distance(adj=None, features=None, labels=None, idx_train=None, device=None):
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

def compute_alpha(n, S_d, d_min=2):
    """
    Approximate the alpha of a power law distribution.
    Parameters
    ----------
    n: int or np.array of int
        Number of entries that are larger than or equal to d_min
    S_d: float or np.array of float
         Sum of log degrees in the distribution that are larger than or equal to d_min
    d_min: int
        The minimum degree of nodes to consider
    Returns
    -------
    alpha: float
        The estimated alpha of the power law distribution
    """

    return n / (S_d - n * np.log(d_min - 0.5)) + 1

def compute_log_likelihood(n, alpha, S_d, d_min=2):
    """
    Compute log likelihood of the powerlaw fit.
    Parameters
    ----------
    n: int
        Number of entries in the old distribution that are larger than or equal to d_min.
    alpha: float
        The estimated alpha of the power law distribution
    S_d: float
         Sum of log degrees in the distribution that are larger than or equal to d_min.
    d_min: int
        The minimum degree of nodes to consider
    Returns
    -------
    float: the estimated log likelihood
    """

    return n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * S_d

def filter_chisquare(ll_ratios, delta_cutoff=0.004):
    return ll_ratios < delta_cutoff

def is_degree_unnoticeable(adj_orig, adj_new, d_min=2):


    degrees_orig = np.asarray(adj_orig.sum(axis=1)).reshape(-1)
    degrees_new = np.asarray(adj_new.sum(axis=1)).reshape(-1)

    log_degree_sum_orig = np.sum(np.log(degrees_orig[degrees_orig >= d_min]))
    log_degree_sum_new = np.sum(np.log(degrees_new[degrees_new >= d_min]))

    n_orig = np.sum(degrees_orig >= d_min)
    n_new = np.sum(degrees_new >= d_min)

    alpha_orig = compute_alpha(n_orig, log_degree_sum_orig, d_min)
    alpha_new = compute_alpha(n_new, log_degree_sum_new, d_min)

    log_likelihood_orig = compute_log_likelihood(n_orig, alpha_orig, log_degree_sum_orig, d_min)
    log_likelihood_new = compute_log_likelihood(n_new, alpha_new, log_degree_sum_new, d_min)

    alpha_combined = compute_alpha(n_orig + n_new, log_degree_sum_orig + log_degree_sum_new, d_min)
    ll_combined = compute_log_likelihood(n_orig + n_new, alpha_combined, log_degree_sum_orig + log_degree_sum_new, d_min)

    ll_ratios = -2 * ll_combined + 2 * (log_likelihood_orig + log_likelihood_new)
    
    return filter_chisquare(ll_ratios), alpha_orig, alpha_new, ll_ratios

def is_difference_significant(p_value, threshold=0.05):
    if p_value is None:
        return False
    else:
        return p_value < threshold


def main():
    df_path = 'reports/eval/degre_noticeability.csv'
    datasets = ['citeseer', 'cora', 'cora_ml', 'polblogs', 'pubmed']
    for dataset in datasets:
        for attack, model_builder, model_name in zip(attacks,model_builders, model_names):
            data = Dataset(root='/tmp/', name=dataset)
            
            # adj,_,_ = preprocess(data.adj, data.features, data.labels, preprocess_adj=False, sparse=True, device=torch.device("cuda" if cuda else "cpu"))
            # acc = test(adj, data, cuda, pre_test_data)
            # row = {'dataset':dataset, 'attack':'Clean', 'seed':None, 'acc':acc}
            # print(row)
            # df = df.append(row, ignore_index=True)
            
            G_orig = nx.from_scipy_sparse_matrix(data.adj)
            ccoefs_orig = np.array(list(nx.clustering(G_orig, nodes=G_orig.nodes, weight=None).values()))
            
            for perturbation_rate in [0.01,0.05, 0.025, 0.10,0.15,0.20]:
                for seed in range(5):
                    modified_adj, elapsed = apply_perturbation(model_builder, attack, data, perturbation_rate, cuda, seed)
                    degre_noticeability, alpha_orig, alpha_new, ll_ratios = is_degree_unnoticeable(data.adj, modified_adj)
                    G_modified = nx.from_scipy_sparse_matrix(modified_adj)
                    ccoefs_modified = np.array(list(nx.clustering(G_modified, nodes=G_orig.nodes, weight=None).values()))
                    try:
                        _, p_value_ccoefs_difference = wilcoxon(ccoefs_orig - ccoefs_modified)
                    except:
                        p_value_ccoefs_difference = None
                        
                    row = {'dataset':dataset, 'attack':model_name, 'seed':seed, 'perturbation_rate':perturbation_rate,'elapsed':elapsed,
                           'is_degree_unnoticeable':degre_noticeability, 'alpha_original':alpha_orig, 'alpha_modified':alpha_new, 'final_test_statistic':ll_ratios,
                           'mean_clustering_coef_orig':np.mean(ccoefs_orig), 'mean_clustering_coef_modified':np.mean(ccoefs_modified), 
                           'ccoef_difference_unnoticeable':not is_difference_significant(p_value_ccoefs_difference), 'p_value_ccoefs_difference':p_value_ccoefs_difference}
                    print(row)
                    cdf = pd.DataFrame()
                    if os.path.exists(df_path):
                        cdf = pd.read_csv(df_path)
                    cdf = cdf.append(row, ignore_index=True)
                    cdf.to_csv(df_path,index=False)


# The following lists should be correspondent
attacks = [
    attack_random,
    attack_dice,
    # attack_structack_fold, 
    # attack_structack_only_distance,
    attack_structack_distance,
    attack_mettaack,
]
model_names = [
    'Random',
    'DICE',
    # 'StructackGreedyFold', # this is StructackDegree in the paper
    # 'StructackOnlyDistance', # this is StructackDistance in the paper
    'StructackDistance', # this is Structack in the paper
    'Metattack',
]
model_builders = [
    build_random,
    build_dice,
    # build_structack_fold,
    # build_structack_only_distance,
    build_structack_distance,
    build_mettack,
]
cuda = torch.cuda.is_available()

if __name__ == '__main__':
    main()