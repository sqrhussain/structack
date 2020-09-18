import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from structack.structack import StructackOneEnd, StructackBothEnds
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse

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
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
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

data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)

perturbations = int(args.ptb_rate * (adj.sum()//2))
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)



def test(adj):
    ''' test on GCN '''

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
    print(f'Accuracy on the clean graph: {test(adj)}')
    if args.percentile_step is None:
        model = StructackBothEnds(degree_percentile_range=[args.frm1,args.to1,args.frm2,args.to2])
        model.attack(adj, perturbations)
        modified_adj = model.modified_adj
        # modified_features = model.modified_features
        accs = []
        for i in range(10):
            model.attack(adj, perturbations)
            modified_adj = model.modified_adj
            # modified_features = model.modified_features
            accs.append(test(modified_adj))
        print(f'percentile [{args.frm1:.2f},{args.to1:.2f}] - [{args.frm2:.2f},{args.to2:.2f}]: {np.mean(accs):.4f} +- {np.std(accs):.2f}')
    else:
        for frm1 in np.arange(0,1,args.percentile_step):
            to1 = frm1 + args.percentile_step
            print(f'percentile [{frm1:.2f},{to1:.2f}]:')
            for frm2 in np.arange(0,1,args.percentile_step):
                to2 = frm2 + args.percentile_step
                model = StructackBothEnds(degree_percentile_range=[frm1,to1,frm2,to2])
                accs = []
                for i in range(10):
                    model.attack(adj, perturbations)
                    modified_adj = model.modified_adj
                    # modified_features = model.modified_features
                    accs.append(test(modified_adj))
                print(f'percentile [{frm2:.2f},{to2:.2f}]: {np.mean(accs):.4f} +- {np.std(accs):.2f}')


if __name__ == '__main__':
    main()
