
import torch
import torch.nn as nn
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
import structack.compare_to_baselines as st

def main():

    ptb_rate_path = 'reports/interim/structack_max_perturbation_rates.csv'
    df_ptb_rate = pd.read_csv(ptb_rate_path)
    ptb_rate_map = {}
    for idx,row in df_ptb_rate.iterrows():
        ptb_rate_map[f"{row['dataset']}+{row['selection']}+{row['connection']}"] = row['max_perturbation_rate']
    
    df_path = 'reports/eval/comb_acc_ptb_rate_threshold.csv'

    selection_options = [
                # [ns.get_random_nodes,'random'],
                [ns.get_nodes_with_lowest_degree,'degree'],
                [ns.get_nodes_with_lowest_pagerank,'pagerank'],
                [ns.get_nodes_with_lowest_eigenvector_centrality,'eigenvector'],
                [ns.get_nodes_with_lowest_betweenness_centrality,'betweenness'],
                [ns.get_nodes_with_lowest_closeness_centrality,'closeness'],
            ]

    connection_options = [
                # [nc.random_connection,'random'],
                [nc.community_hungarian_connection,'community'],
                [nc.distance_hungarian_connection,'distance'],
                [nc.katz_hungarian_connection,'katz'],
            ]

    datasets = ['citeseer', 'cora', 'cora_ml', 'polblogs', 'pubmed']
    # datasets = ['cora']
    split_seeds = 5
    gcn_seeds = 5
    for selection, selection_name in selection_options:
        for connection, connection_name in connection_options:
            for dataset in datasets:
                data = Dataset(root='/tmp/', name=dataset)
                print(f'attack [{selection_name}]*[{connection_name}]')
                if f'{dataset}+{selection_name}+{connection_name}' not in ptb_rate_map:
                    continue
                perturbation_rate = ptb_rate_map[f'{dataset}+{selection_name}+{connection_name}']
                modified_adj, elapsed = st.apply_structack(build_custom(selection, connection), st.attack_structack, data, perturbation_rate, cuda and (dataset!='pubmed'), seed=0)
                for split_seed in range(split_seeds):
                    np.random.seed(split_seed)
                    torch.manual_seed(split_seed)
                    if cuda:
                        torch.cuda.manual_seed(split_seed)
                    
                    # reload the dataset with a different split (WARNING: this doesn't work for attack methods which depend on the split)
                    data = Dataset(root='/tmp/', name=dataset)

                    for seed in range(gcn_seeds):

                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        if cuda:
                            torch.cuda.manual_seed(seed)

                        acc = st.test_gcn(modified_adj, data, cuda, st.pre_test_data)
                        row = {'dataset':dataset, 'selection':selection_name, 'connection':connection_name,
                                'gcn_seed':seed, 'acc':acc, 'perturbation_rate':perturbation_rate,'elapsed':elapsed,
                                'split_seed':split_seed}
                        print(row)
                        cdf = pd.DataFrame()
                        if os.path.exists(df_path):
                            cdf = pd.read_csv(df_path)
                        cdf = cdf.append(row, ignore_index=True)
                        cdf.to_csv(df_path,index=False)

cuda = torch.cuda.is_available()

if __name__ == '__main__':
    main()