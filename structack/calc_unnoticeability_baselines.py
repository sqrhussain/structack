import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.global_attack import DICE, Random, Metattack, PGDAttack, MinMax
from structack.structack import StructackBase, build_custom
from structack.structack import StructackDegreeRandomLinking, StructackDegree, StructackDegreeDistance,StructackDistance
from structack.structack import StructackEigenvectorCentrality, StructackBetweennessCentrality, StructackClosenessCentrality
from structack.structack import StructackPageRank, StructackKatzSimilarity, StructackCommunity
import structack.node_selection as ns
import structack.node_connection as nc
# from structack.calc_unnoticeability import *
import pandas as pd
import time
import os
from scipy import stats
from scipy.stats import wilcoxon
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import norm
import networkx as nx
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run structack.")
    parser.add_argument('--datasets', nargs='+', default=['citeseer', 'cora', 'cora_ml', 'polblogs', 'pubmed'], help='List of datasets to evaluate.')
    parser.add_argument('--output', nargs='?', default='reports/eval/comb_acc_eval_noticeability.csv', help='Evaluation results output filepath.')
    parser.add_argument('--approach_type', nargs='?', default='structack', help='Type of approaches to run [baseline/structack].')
    parser.add_argument('--ptb', nargs='+', type=float, default=[0.005, 0.0075, 0.01, 0.025,0.05, 0.075, 0.10, 0.15, 0.20])
    return parser.parse_args()

def postprocess_adj(adj):
    # adj = normalize_adj(adj)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    if type(adj) is torch.Tensor:
        adj = to_scipy(adj)
    print(f'TYPE: {type(adj)}')
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
    return postprocess_adj(model.modified_adj)


def attack_minmax(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled):
    model.attack(features, adj, labels, idx_train, n_perturbations)
    return postprocess_adj(model.modified_adj)


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
#     modified_adj = modified_adj.to(device)
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

    if model_builder in [build_mettack, build_pgd, build_minmax]:
        adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

    # build the model
    model = model_builder(adj, features, labels, idx_train, device)
        
    tick = time.time()
    # perform the attack
    modified_adj = attack(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled)
    elapsed = time.time() - tick

    # modified_adj = to_scipy(modified_adj)
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

def calc_wilcoxon(orig, mod):
    try:
        _, p_value = wilcoxon(orig - mod)
    except:
        p_value = None
    return p_value

def calc_relative_change(orig, mod):
    denominator = np.array(orig)
    denominator[denominator == 0] = np.nan
    return np.nan_to_num(np.abs((mod-orig)/denominator))

def calc_relative_diff(orig, mod, denominator_type, lala):
    if denominator_type == 'max':
        denominator = np.array([max(z) for z in map(lambda x, y:(x,y), orig, mod)])
        denominator[denominator == 0] = np.nan
        return np.nan_to_num(np.abs((mod-orig)/denominator))
    elif denominator_type == 'min':
        denominator = np.array([min(z) for z in map(lambda x, y:(x,y), orig, mod)])
        denominator[denominator == 0] = np.nan
        return np.nan_to_num(np.abs((mod-orig)/denominator))
    elif denominator_type == 'mean':
        denominator = (mod+orig)/2
        denominator[denominator == 0] = np.nan
        return np.nan_to_num(np.abs((mod-orig)/denominator))
    
def extend_row_with_noticeability(row, G_orig, degree_centralities_orig, ccoefs_orig, adj, modified_adj):
    G_modified = nx.from_scipy_sparse_matrix(modified_adj)
    degree_centralities_modified = np.array(list(nx.degree_centrality(G_modified).values()))
    ccoefs_modified = np.array(list(nx.clustering(G_modified, nodes=G_orig.nodes, weight=None).values()))
                    
    p_value_degree_centralities_wilcoxon = calc_wilcoxon(degree_centralities_orig, degree_centralities_modified)
    p_value_ccoefs_wilcoxon = calc_wilcoxon(ccoefs_orig, ccoefs_modified)
                        
    relative_degree_change = calc_relative_change(degree_centralities_orig, degree_centralities_modified)
    relative_ccoefs_change = calc_relative_change(ccoefs_orig, ccoefs_modified)
                    
    relative_degree_diff_min = calc_relative_diff(degree_centralities_orig, degree_centralities_modified, 'min', 'degree')
    relative_degree_diff_max = calc_relative_diff(degree_centralities_orig, degree_centralities_modified, 'max', 'degree')
    relative_degree_diff_mean = calc_relative_diff(degree_centralities_orig, degree_centralities_modified, 'mean', 'degree')
                    
    relative_ccoefs_diff_min = calc_relative_diff(ccoefs_orig, ccoefs_modified, 'min', 'ccoefs')
    relative_ccoefs_diff_max = calc_relative_diff(ccoefs_orig, ccoefs_modified, 'max', 'ccoefs')
    relative_ccoefs_diff_mean = calc_relative_diff(ccoefs_orig, ccoefs_modified, 'mean', 'ccoefs')
                    
    dc_kstest_statistic, dc_kstest_pvalue = stats.ks_2samp(degree_centralities_orig, degree_centralities_modified)
    cc_kstest_statistic, cc_kstest_pvalue = stats.ks_2samp(ccoefs_orig, ccoefs_modified)
    
    print(len(G_orig.nodes))
    print(len(G_modified.nodes))
    print(len(G_orig.edges))
    print(len(G_modified.edges))
    print(abs(len(G_orig.edges)-len(G_modified.edges)))
    
    row = {
        'dataset':row['dataset'], 
        'attack':row['attack'],
        'attack_seed':row['attack_seed'],
        'split_seed':row['split_seed'],
        'perturbation_rate':row['perturbation_rate'],
        'elapsed':row['elapsed'],
        'edge_count_diff':abs(len(G_orig.edges)-len(G_modified.edges)),
        
        'mean_degree_centralities_orig':np.mean(degree_centralities_orig), 
        'mean_degree_centralities_modified':np.mean(degree_centralities_modified), 
        'p_value_degree_centralities_wilcoxon':p_value_degree_centralities_wilcoxon,
        
        'mean_clustering_coef_orig':np.mean(ccoefs_orig), 
        'mean_clustering_coef_modified':np.mean(ccoefs_modified), 
        'p_value_ccoefs_wilcoxon':p_value_ccoefs_wilcoxon,
        
        'degree_centralities_kstest_statistic':dc_kstest_statistic,
        'degree_centralities_kstest_pvalue':dc_kstest_pvalue,
        
        'ccoefs_kstest_statistic':cc_kstest_statistic, 
        'ccoefs_kstest_pvalue':cc_kstest_pvalue,
        
        'mean_relative_degree_change_all_nodes':np.mean(relative_degree_change),
        'mean_relative_degree_change_perturbed_nodes':np.nanmean(np.where(relative_degree_change!=0,relative_degree_change,np.nan),0),
        'mean_relative_ccoefs_change_all_nodes':np.mean(relative_ccoefs_change),
        'mean_relative_ccoefs_change_perturbed_nodes':np.nanmean(np.where(relative_ccoefs_change!=0,relative_ccoefs_change,np.nan),0),
        
        'degree_assortativity_orig':nx.degree_assortativity_coefficient(G_orig),
        'degree_assortativity_modified':nx.degree_assortativity_coefficient(G_modified),
        'relative_degree_assortativity_change':calc_relative_change(nx.degree_assortativity_coefficient(G_orig), nx.degree_assortativity_coefficient(G_modified)),
        
        'mean_relative_degree_diff_min_all_nodes':np.mean(relative_degree_diff_min),
        'mean_relative_degree_diff_min_perturbed_nodes':np.nanmean(np.where(relative_degree_diff_min!=0,relative_degree_diff_min,np.nan),0),
        'mean_relative_degree_diff_max_all_nodes':np.mean(relative_degree_diff_max),
        'mean_relative_degree_diff_max_perturbed_nodes':np.nanmean(np.where(relative_degree_diff_max!=0,relative_degree_diff_max,np.nan),0),
        'mean_relative_degree_diff_mean_all_nodes':np.mean(relative_degree_diff_mean),
        'mean_relative_degree_diff_mean_perturbed_nodes':np.nanmean(np.where(relative_degree_diff_mean!=0,relative_degree_diff_mean,np.nan),0),
        
        'mean_relative_ccoefs_diff_min_all_nodes':np.mean(relative_ccoefs_diff_min),
        'mean_relative_ccoefs_diff_min_perturbed_nodes':np.nanmean(np.where(relative_ccoefs_diff_min!=0,relative_ccoefs_diff_min,np.nan),0),
        'mean_relative_ccoefs_diff_max_all_nodes':np.mean(relative_ccoefs_diff_max),
        'mean_relative_ccoefs_diff_max_perturbed_nodes':np.nanmean(np.where(relative_ccoefs_diff_max!=0,relative_ccoefs_diff_max,np.nan),0),
        'mean_relative_ccoefs_diff_mean_all_nodes':np.mean(relative_ccoefs_diff_mean),
        'mean_relative_ccoefs_diff_mean_perturbed_nodes':np.nanmean(np.where(relative_ccoefs_diff_mean!=0,relative_ccoefs_diff_mean,np.nan),0)}
    return row


def main(args):
    datasets = args.datasets
    df_path = args.output
    perturbation_rates = args.ptb
    
    attacks = [
        # [attack_random, 'Random', build_random],
#         [attack_dice, 'DICE', build_dice],
#         [attack_mettaack, 'Metattack', build_mettack],
        # [attack_pgd, 'PGD', build_pgd],
        [attack_minmax, 'MinMax', build_minmax],
    ]
    for dataset in datasets:
        for attack, model_name, model_builder in attacks:
            print('attack ' + model_name)
            for split_seed in range(5):
                np.random.seed(split_seed)
                torch.manual_seed(split_seed)
                if cuda:
                    torch.cuda.manual_seed(split_seed)
                data = Dataset(root='/tmp/', name=dataset)
                G_orig = nx.from_scipy_sparse_matrix(data.adj)
                degree_centralities_orig = np.array(list(nx.degree_centrality(G_orig).values()))
                ccoefs_orig = np.array(list(nx.clustering(G_orig, nodes=G_orig.nodes, weight=None).values()))
                for perturbation_rate in perturbation_rates:
                    for attack_seed in range(1 if model_name=='DICE' else 5):
                        modified_adj, elapsed = apply_perturbation(model_builder, attack, data, perturbation_rate, cuda and (dataset!='pubmed'), attack_seed)
                        print(type(modified_adj))
                        row = {
                            'dataset':dataset, 
                            'attack':model_name, 
                            'perturbation_rate':perturbation_rate,
                            'elapsed':elapsed, 
                            'attack_seed' :attack_seed,
                            'split_seed':split_seed}
                        row = extend_row_with_noticeability(row, G_orig, degree_centralities_orig, ccoefs_orig, data.adj, modified_adj)
                        print(row)
                        cdf = pd.DataFrame()
                        if os.path.exists(df_path):
                            cdf = pd.read_csv(df_path)
                        cdf = cdf.append(row, ignore_index=True)
                        cdf.to_csv(df_path,index=False)


def combination(args):
    datasets = args.datasets
    df_path = args.output
    
    selection_options = [
                [ns.get_random_nodes,'random'],
                [ns.get_nodes_with_lowest_degree,'degree'],
                [ns.get_nodes_with_lowest_pagerank,'pagerank'],
                [ns.get_nodes_with_lowest_eigenvector_centrality,'eigenvector'],
                [ns.get_nodes_with_lowest_betweenness_centrality,'betweenness'],
                [ns.get_nodes_with_lowest_closeness_centrality,'closeness'],
            ]

    connection_options = [
                [nc.random_connection,'random'],
                [nc.community_hungarian_connection,'community'],
                [nc.distance_hungarian_connection,'distance'],
                [nc.katz_hungarian_connection,'katz'],
            ]
    
        
    for dataset in datasets:
        data = Dataset(root='/tmp/', name=dataset)
        G_orig = nx.from_scipy_sparse_matrix(data.adj)
        degree_centralities_orig = np.array(list(nx.degree_centrality(G_orig).values()))
        ccoefs_orig = np.array(list(nx.clustering(G_orig, nodes=G_orig.nodes, weight=None).values()))
        
        for selection, selection_name in selection_options:
            for connection, connection_name in connection_options:
                print(f'attack [{selection_name}]*[{connection_name}]')
                for perturbation_rate in [0.005, 0.0075, 0.01, 0.025,0.05, 0.075, 0.10, 0.15, 0.20]:
                    for seed in range(5 if (selection_name == 'random' or connection_name == 'random') else 1):
                        modified_adj, elapsed = apply_structack(build_custom(selection, connection), attack_structack, data, perturbation_rate, cuda and (dataset!='pubmed'), seed=seed)
                        
                        # reload the dataset with a different split (WARNING: this doesn't work for attack methods which depend on the split)
                        data = Dataset(root='/tmp/', name=dataset)

                        row = {
                            'dataset':dataset, 
                            'selection':selection_name, 
                            'connection':connection_name,
                            'gcn_seed':seed, 
                            'perturbation_rate':perturbation_rate,
                            'elapsed':elapsed}
                        row = extend_row_with_noticeability(row, G_orig, degree_centralities_orig, ccoefs_orig, data.adj, modified_adj)
                        print(row)
                        cdf = pd.DataFrame()
                        if os.path.exists(df_path):
                            cdf = pd.read_csv(df_path)
                        cdf = cdf.append(row, ignore_index=True)
                        cdf.to_csv(df_path,index=False)


cuda = torch.cuda.is_available()

if __name__ == '__main__':
    args = parse_args()
    if args.approach_type == 'baseline':
        main(args)
    elif args.approach_type == 'structack':
        combination(args)