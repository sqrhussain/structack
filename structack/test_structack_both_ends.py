import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from structack.structack import StructackBothEnds
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
import pandas as pd
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, nargs='+', default=['citeseer'],
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--percentile_step', type=float, default=None, help='If None, [frm,to] is taken instead')
parser.add_argument('--frm1', type=float, default=0,help='Percentile from')
parser.add_argument('--to1', type=float, default=1,help='Percentile to')
parser.add_argument('--frm2', type=float, default=0,help='Percentile from')
parser.add_argument('--to2', type=float, default=1,help='Percentile to')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)





def test(data,adj):
    ''' test on GCN '''

    _, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train) # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output.cpu()
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test.item()),
    #       "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()

def main():
    # print(f'Accuracy on the clean graph: {test(adj)}')
    if args.percentile_step is None:
        for dataset in datasets:
            data = Dataset(root='/tmp/', name=dataset, setting='nettack')
            perturbations = int(args.ptb_rate * (data.adj.sum()//2))

            model = StructackBothEnds(degree_percentile_range=[args.frm1,args.to1,args.frm2,args.to2])
            model.attack(data.adj, perturbations)
            modified_adj = model.modified_adj
            # modified_features = model.modified_features
            accs = []
            for seed_i in range(10):
                model.attack(data.adj, perturbations)
                modified_adj = model.modified_adj
                # modified_features = model.modified_features
                accs.append(test(data,modified_adj))
            print(f'percentile [{args.frm1:.2f},{args.to1:.2f}] - [{args.frm2:.2f},{args.to2:.2f}]: {np.mean(accs):.4f} +- {np.std(accs):.2f}')
    else:
        df_path = 'reports/eval/both_ends.csv'
        for dataset in args.dataset:
            data = Dataset(root='/tmp/', name=dataset, setting='nettack')
            perturbations = int(args.ptb_rate * (data.adj.sum()//2))
            print(f'Accuracy on the clean graph: {test(data,data.adj)}')
            for frm1 in np.arange(0,1,args.percentile_step):
                to1 = frm1 + args.percentile_step
                print(f'percentile [{frm1:.2f},{to1:.2f}]:')
                for frm2 in np.arange(0,to1,args.percentile_step):
                    print(f'===={frm1},{frm2}====')
                    to2 = frm2 + args.percentile_step
                    model = StructackBothEnds(degree_percentile_range=[frm1,to1,frm2,to2])
                    accs = []
                    for seed_i in range(10):
                        tick = time.time()
                        model.attack(data.adj, perturbations)
                        elapsed = time.time()-tick
                        modified_adj = model.modified_adj
                        # modified_features = model.modified_features
                        acc = test(data,modified_adj)
                        accs.append(acc)
                        cdf = pd.DataFrame()
                        if os.path.exists(df_path):
                            cdf = pd.read_csv(df_path)
                        row = {'dataset':dataset, 'attack':model.__class__.__name__,
                            'seed':seed_i, 'acc':acc, 'perturbation_rate':args.ptb_rate,'elapsed':elapsed,
                            'frm1':frm1,'to1':to1,'frm2':frm2,'to2':to2}
                        print(row)
                        cdf = cdf.append(row, ignore_index=True)
                        cdf.to_csv(df_path,index=False)

                    print(f'percentile [{frm2:.2f},{to2:.2f}]: {np.mean(accs):.4f} +- {np.std(accs):.2f}')


if __name__ == '__main__':
    main()
