'''
    If you would like to reproduce the performance of the paper,
    please refer to https://github.com/ChandlerBang/Pro-GNN
'''


import time
import argparse
import numpy as np
import torch
from deeprobust.graph.defense import GCN, ProGNN, GCNSVD, GCNJaccard
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import preprocess
from structack.compare_to_baselines import *
from deeprobust.graph.utils import *


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--only_gcn', action='store_true',
        default=False, help='test the performance of gcn without other components')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
        choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--attack', type=str, default='meta',
        choices=['no', 'meta', 'random', 'nettack'])
parser.add_argument('--ptb_rate', type=float, default=0.05, help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=400, help='Number of epochs to train.')
parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
parser.add_argument('--lambda_', type=float, default=0, help='weight of feature smoothing')
parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
parser.add_argument('--symmetric', action='store_true', default=False,
            help='whether use symmetric matrix')
parser.add_argument('--defense')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.ptb_rate == 0:
    args.attack = "no"

print(args)

np.random.seed(15) # Here the random seed is to split the train/val/test data, we need to set the random seed to be the same as that when you generate the perturbed graph

# data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
# adj, features, labels = data.adj, data.features, data.labels
# idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

# if args.attack == 'no':
#     perturbed_adj = adj

# if args.attack == 'random':
#     from deeprobust.graph.global_attack import Random
#     attacker = Random()
#     n_perturbations = int(args.ptb_rate * (adj.sum()//2))
#     perturbed_adj = attacker.attack(adj, n_perturbations, type='add')

# if args.attack == 'meta' or args.attack == 'nettack':
#     perturbed_data = PrePtbDataset(root='/tmp/',
#             name=args.dataset,
#             attack_method=args.attack,
#             ptb_rate=args.ptb_rate)
#     perturbed_adj = perturbed_data.adj


np.random.seed(args.seed)
torch.manual_seed(args.seed)


def defend(perturbed_adj, data, data_prep=pre_test_data, nhid=16):
    device = torch.device("cuda" if cuda else "cpu")
    # print(type(perturbed_adj))
    # print(is_sparse_tensor(perturbed_adj))
    # perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, device=device)

    if args.defense == 'prognn':
        features, labels, idx_train, idx_val, idx_test = data_prep(data,device)
        model = GCN(nfeat=features.shape[1],
                    nhid=nhid,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout, device=device)

        print('=== testing Pro-GNN on perturbed graph ===')
        prognn = ProGNN(model, args, device)
        prognn.fit(features, perturbed_adj.to_dense(), labels, idx_train, idx_val)
        return prognn.test(features, labels, idx_test).item()
    elif args.defense == 'gcn-jaccard':
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        perturbed_adj = to_scipy(perturbed_adj)
        model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max()+1,
                        nhid=nhid, device=device)

        model = model.to(device)

        print('=== testing GCN-Jaccard on perturbed graph ===')
        model.fit(features, perturbed_adj, labels, idx_train, idx_val, threshold=0.01)
        model.eval()
        return model.test(idx_test).item()
    elif args.defense == 'gcn-svd':
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        perturbed_adj = to_scipy(perturbed_adj)
        model = GCNSVD(nfeat=features.shape[1], nclass=labels.max()+1,
                        nhid=16, device=device)

        model = model.to(device)

        print('=== testing GCN-SVD on perturbed graph ===')
        model.fit(features, perturbed_adj, labels, idx_train, idx_val, k=15, verbose=True)
        model.eval()
        return model.test(idx_test).item()

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


def main():
    df_path = 'reports/eval/defense.csv'
    datasets = ['citeseer', 'cora', 'cora_ml', 'polblogs', 'pubmed']
    datasets = ['citeseer']
    for dataset in datasets:
        for attack, model_builder, model_name in zip(attacks,model_builders, model_names):
            data = Dataset(root='/tmp/', name=dataset)
            # adj,_,_ = preprocess(data.adj, data.features, data.labels, preprocess_adj=False, sparse=True, device=torch.device("cuda" if cuda else "cpu"))
            # acc = test(adj, data, cuda, pre_test_data)
            # row = {'dataset':dataset, 'attack':'Clean', 'seed':None, 'acc':acc}
            # print(row)
            # df = df.append(row, ignore_index=True)
            for perturbation_rate in [0.05]: #,0.01,0.10,0.15,0.20]:
                print(f'{dataset} {model_name} {perturbation_rate}')
                for seed in range(10):
                    perturbed_adj, elapsed = apply_perturbation(model_builder, attack, data, perturbation_rate, cuda, seed)
                    acc = defend(perturbed_adj, data)
                    row = {'dataset':dataset, 'attack':model_name, 'defense': args.defense, 'seed':seed, 'acc':acc, 'perturbation_rate':perturbation_rate,'elapsed':elapsed}
                    print(row)
                    cdf = pd.DataFrame()
                    if os.path.exists(df_path):
                        cdf = pd.read_csv(df_path)
                    cdf = cdf.append(row, ignore_index=True)
                    cdf.to_csv(df_path,index=False)



if __name__ == '__main__':
    main()