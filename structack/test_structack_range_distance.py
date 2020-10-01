import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from structack.structack import StructackRangeDistance
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
parser.add_argument('--ptb_rate', type=float, nargs='+', default=[0.05],  help='pertubation rate')
parser.add_argument('--percentile_step', type=float, default=None, help='If None, [frm,to] is taken instead')
parser.add_argument('--frm', type=float, default=0,help='Percentile from')
parser.add_argument('--to', type=float, default=1,help='Percentile to')

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
    df_path = 'reports/eval/distance_range.csv'
    for dataset in args.dataset:
        data = Dataset(root='/tmp/', name=dataset, setting='nettack')
        print(f'Accuracy on the clean graph: {test(data,data.adj)}')
        for ptb_rate in args.ptb_rate:
            perturbations = int(ptb_rate * (data.adj.sum()//2))
            if args.percentile_step is None:
                model = StructackRangeDistance(distance_percentile_range=[args.frm,args.to])
                model.attack(adj, perturbations)
                modified_adj = model.modified_adj
                # modified_features = model.modified_features
                test(data,modified_adj)
            else:
                for frm in np.arange(0,1,args.percentile_step):
                    print(f'===={frm}====')
                    to = frm + args.percentile_step
                    model = StructackRangeDistance(distance_percentile_range=[frm,to])
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
                            'seed':seed_i, 'acc':acc, 'perturbation_rate':ptb_rate,'elapsed':elapsed,
                            'frm':frm,'to':to, 'mean_distance':model.mean_distance}
                        print(row)
                        cdf = cdf.append(row, ignore_index=True)
                        cdf.to_csv(df_path,index=False)

                    print(f'percentile [{frm:.2f},{to:.2f}]: {np.mean(accs):.4f} +- {np.std(accs):.2f}')

    # # if you want to save the modified adj/features, uncomment the code below
    # model.save_adj(root='./', name=f'mod_adj')
    # model.save_features(root='./', name='mod_features')

if __name__ == '__main__':
    main()
