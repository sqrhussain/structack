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
from structack.structack import StructackDegree, StructackDistance, StructackDegreeDistance

import time
import os

import pandas as pd

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

attacks = {'random':Random(), 'dice':DICE(), 'sfold':StructackDegree(), 'sdist':StructackDistance(),
            'sdegdist':StructackDegreeDistance()}
model_names = {
    'clean':'Clean',
    'random' : 'Random',
    'dice':'DICE',
    # 'StructackGreedyRandom',
    # 'StructackOneEnd',
    # 'StructackBothEnds',
    'sfold' : 'StructackGreedyFold',
    'sdist' : 'StructackOnlyDistance',
    'sdegdist': 'StructackDistance',
    'metattack' : 'Metattack',
}

basic_attacks = 'random sfold sdist'.split()


def attack_dice(model, adj, labels, n_perturbations):
    model.attack(adj, labels, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)

def attack_random(model, adj, labels, n_perturbations):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return modified_adj

def attack_structack_fold(model, adj, labels, n_perturbations):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return modified_adj

def attack_structack_distance(model, adj, labels, n_perturbations):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return modified_adj

def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--attack', default='clean')
    parser.add_argument('--ptb_rate', type=float, default=0.01)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset_name = 'ogbn-arxiv'
    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()

    if args.attack!='clean':
        # print(data)
        print(data.adj_t)
        attack_model = attacks[args.attack]
        adj = data.adj_t.to_torch_sparse_coo_tensor()
        adj = to_scipy(adj)
        n = adj.sum()//2
        print(n)
        n_perturbations = int(args.ptb_rate * (adj.sum()//2))
        print(f'n_perturbations = {n_perturbations}')
        
        tick = time.time()
        if args.attack=='dice':
            attack_model.attack(adj, data.y.numpy(), n_perturbations)
        else:
            attack_model.attack(adj, n_perturbations)
        modified_adj = attack_model.modified_adj
        elapsed = time.time() - tick
        data.adj_t = SparseTensor.from_torch_sparse_coo_tensor(sparse_mx_to_torch_sparse_tensor(modified_adj))
    else:
        elapsed = 0

    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    # logger = Logger(args.runs, args)

    df_path = 'reports/eval/ogb.csv'

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            # logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
        row = {'dataset':dataset_name, 'attack':model_names[args.attack], 'seed':run, 'acc':test_acc, 'perturbation_rate':args.ptb_rate,'elapsed':elapsed}
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
