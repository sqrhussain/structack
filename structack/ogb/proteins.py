import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from torch_sparse import SparseTensor

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

# from logger import Logger

from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor, to_scipy
from deeprobust.graph.global_attack import DICE, Random
from structack.structack import StructackDegree, StructackDegreeDistance

import time
import os

import pandas as pd


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


def train(model, data, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = criterion(out, data.y[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    y_pred = model(data.x, data.adj_t)

    train_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


attacks = {'random':Random(), 'dice':DICE(), 'sfold':StructackDegree(), 'sdist':StructackDegreeDistance()}
model_names = {
    'clean':'Clean',
    'random' : 'Random',
    'dice':'DICE',
    # 'StructackGreedyRandom',
    # 'StructackOneEnd',
    # 'StructackBothEnds',
    'sfold' : 'StructackGreedyFold',
    'sdist' : 'StructackDistance',
    'metattack' : 'Metattack',
}

basic_attacks = 'random sfold sdist'.split()

def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--attack', default='clean')
    parser.add_argument('--ptb_rate', type=float, default=0.01)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset_name = 'ogbn-proteins'
    dataset = PygNodePropPredDataset(name='ogbn-proteins',
                                     transform=T.ToSparseTensor())
    data = dataset[0]


    if args.attack!='clean':
        print(data.adj_t)
        attack_model = attacks[args.attack]
        adj = data.adj_t.to_torch_sparse_coo_tensor()
        print(adj.shape)
        adj = to_scipy(adj)
        n = adj.sum()//2
        print(n)
        n_perturbations = int(args.ptb_rate * (adj.sum()//2))
        print(f'n_perturbations = {n_perturbations}')
        
        tick = time.time()
        attack_model.attack(adj, n_perturbations)
        elapsed = time.time() - tick
        modified_adj = attack_model.modified_adj
        data.adj_t = SparseTensor.from_torch_sparse_coo_tensor(sparse_mx_to_torch_sparse_tensor(modified_adj))
    else:
        elapsed = 0

    # Move edge features to node features.
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels, 112,
                     args.num_layers, args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels, 112,
                    args.num_layers, args.dropout).to(device)

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    data = data.to(device)

    evaluator = Evaluator(name='ogbn-proteins')
    # logger = Logger(args.runs, args)

    df_path = 'reports/eval/ogb.csv'

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)

            if epoch % args.eval_steps == 0:
                result = test(model, data, split_idx, evaluator)
                # logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_rocauc:.2f}%, '
                          f'Valid: {100 * valid_rocauc:.2f}% '
                          f'Test: {100 * test_rocauc:.2f}%')
        row = {'dataset':dataset_name, 'attack':model_names[args.attack], 'seed':run, 'acc':test_rocauc, 'perturbation_rate':args.ptb_rate,'elapsed':elapsed}
        print(row)
        cdf = pd.DataFrame()
        if os.path.exists(df_path):
            cdf = pd.read_csv(df_path)
        cdf = cdf.append(row, ignore_index=True)
        cdf.to_csv(df_path,index=False)

        # logger.print_statistics(run)
    # logger.print_statistics()


if __name__ == "__main__":
    main()
