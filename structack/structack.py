import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from deeprobust.graph import utils
from deeprobust.graph.global_attack import BaseAttack
import networkx as nx
from scipy.optimize import linear_sum_assignment
import time
# from dgl.traversal import bfs_nodes_generator
from structack.bfs import bfs

class StructackBase(BaseAttack):
    def __init__(self):
        super(StructackBase, self).__init__(None, None, attack_structure=True, attack_features=False, device='cpu')
        self.modified_adj = None

    def get_nodes_with_degree_percentile(self, G, frm, to):
        nodes = sorted(G.degree, key=lambda x: x[1])
        length = len(nodes)
        frm = int(length*frm)
        to =  int(length*to +1)
        nodes = nodes[frm:to]
        return [x[0] for x in nodes]

    def get_nodes_with_lowest_degree(self, G, n):
        nodes = sorted(G.degree, key=lambda x: x[1])
        nodes = nodes[:n]
        return [x[0] for x in nodes]

    def get_purturbed_adj(self, adj, n_perturbations):
        pass

    def attack(self, ori_adj, n_perturbations):
        self.modified_adj = self.get_purturbed_adj(ori_adj,n_perturbations)

class StructackOneEnd(StructackBase):
    def __init__(self, degree_percentile_range=[0,1]):
        super(StructackOneEnd, self).__init__()
        self.frm, self.to = degree_percentile_range
    def get_purturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
        
        # perturb
        node_pool = self.get_nodes_with_degree_percentile(graph,self.frm,self.to)
        rows = np.random.choice(node_pool,n_perturbations)
        cols = np.random.choice(graph.nodes(),n_perturbations)
        edges = [[u,v] for u,v in zip(rows, cols)]
        graph.add_edges_from(edges)

        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj

class StructackBothEnds(StructackBase):
    def __init__(self, degree_percentile_range=[0,1,0,1]):
        super(StructackBothEnds, self).__init__()
        self.frm1, self.to1, self.frm2, self.to2 = degree_percentile_range
    def get_purturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
        
        # perturb
        rows = np.random.choice(self.get_nodes_with_degree_percentile(graph,self.frm1,self.to1),n_perturbations)
        cols = np.random.choice(self.get_nodes_with_degree_percentile(graph,self.frm2,self.to2),n_perturbations)
        edges = [[u,v] for u,v in zip(rows, cols)]
        graph.add_edges_from(edges)

        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj

class StructackDegreeRandomLinking(StructackBase):
    def __init__(self):
        super(StructackDegreeRandomLinking, self).__init__()
        self.modified_adj = None

    def get_purturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
        n = adj.shape[0]

        # select nodes
        nodes = self.get_nodes_with_lowest_degree(graph,2*n_perturbations)
        np.random.shuffle(nodes)

        # perturb
        rows = nodes[:n_perturbations]
        cols = nodes[n_perturbations:]
        edges = [[u,v] for u,v in zip(rows, cols)]
        graph.add_edges_from(edges)

        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj



class StructackDegree(StructackBase):
    def __init__(self):
        super(StructackDegree, self).__init__()
        self.modified_adj = None

    def get_purturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
        n = adj.shape[0]

        # select nodes
        nodes = self.get_nodes_with_lowest_degree(graph,2*n_perturbations)

        # perturb
        rows = nodes[:n_perturbations]
        cols = nodes[n_perturbations:]
        edges = [[u,v] for u,v in zip(rows, cols)]
        graph.add_edges_from(edges)

        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj


class StructackDistance(StructackBase):
    def __init__(self):
        super(StructackDistance, self).__init__()
        self.INF = 1e9+7
        self.modified_adj = None


    def get_purturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph)
        n = adj.shape[0]
        tick = time.time()
        # select nodes
        nodes = np.random.choice(graph.nodes(), size=2*n_perturbations, replace=(len(graph.nodes())<2*n_perturbations))

        rows = nodes[:n_perturbations]
        cols = nodes[n_perturbations:]
        print(f'{self.__class__.__name__}: obtained nodes in {time.time()-tick}')


        tick = time.time()
        dgl_graph = dgl.from_networkx(graph)
        e0 = [e[0] for e in graph.edges()]
        e1 = [e[1] for e in graph.edges()]
        dgl_graph.add_edges(e1,e0)
        bfs_nodes_generator(dgl_graph,rows[0])
        print(f'{self.__class__.__name__}: computed SSSP on one node in {time.time()-tick}')
        tick = time.time()
        bfs_nodes = {u:bfs_nodes_generator(dgl_graph,u) for u in rows}
        distance = {u:{v.item():i for i,lvl in enumerate(bfs_nodes[u]) for v in lvl} for u in rows}
        distance = {u:{v:distance[u][v] if v in distance[u] else self.INF for v in cols} for u in rows}
        # distance = {u:nx.single_source_shortest_path_length(graph,u) for u in rows}
        # distance = {u:{v:distance[u][v] for v in cols} for u in rows}
        print(f'{self.__class__.__name__}: computed distance in {time.time()-tick}')

        tick = time.time()
        mtx = np.array([np.array(list(distance[u].values())) for u in distance])

        i_u = {i:u for i,u in enumerate(distance)}
        i_v = {i:v for i,v in enumerate(distance[list(distance.keys())[0]])}

        u,v = linear_sum_assignment(-mtx)
        print(f'{self.__class__.__name__}: computed assignment in {time.time()-tick}')

        tick = time.time()
        edges = [[i_u[i],i_v[j]] for i,j in zip(u,v)]
        graph.add_edges_from(edges)
        print(f'{self.__class__.__name__}: added edges in {time.time()-tick}')

        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj


class StructackRangeDistance(StructackBase):
    def __init__(self, distance_percentile_range=[0,1]):
        super(StructackRangeDistance, self).__init__()
        self.frm, self.to = distance_percentile_range
        self.modified_adj = None


    def get_purturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph)
        n = adj.shape[0]
        tick = time.time()
        # select nodes
        rows = np.random.choice(graph.nodes(), size=n_perturbations, replace=(len(graph.nodes())<2*n_perturbations))

        print(f'{self.__class__.__name__}: obtained nodes in {time.time()-tick}')


        tick = time.time()
        distance = {u:nx.single_source_shortest_path_length(graph,u) for u in rows}

        frm = int(self.frm*n)
        to = int(self.to*n)
        distance = {u:sorted(distance[u].items(),key=lambda x:x[1])[frm:to] for u in distance}
        idx = {u:np.random.choice(len(distance[u])) for u in distance}
        print(f'{self.__class__.__name__}: computed distance in {time.time()-tick}')

        tick = time.time()
        edges = [[u,distance[u][idx[u]][0]] for u in distance]
        distance = [distance[u][idx[u]][1] for u in distance]
        # print(distance)
        # print(edges)
        # exit(0)
        graph.add_edges_from(edges)
        print(f'{self.__class__.__name__}: added edges in {time.time()-tick}')

        # print(distance)
        # print(edges)
        # exit(0)
        self.mean_distance = np.mean(distance)
        print(f'mean distance = {self.mean_distance:.2f}')

        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj


class StructackDegreeDistance(StructackBase):
    def __init__(self):
        super(StructackDegreeDistance, self).__init__()
        self.INF = 1e9+7
        self.modified_adj = None

    def get_purturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph)
        n = adj.shape[0]
        tick = time.time()
        # select nodes
        nodes = self.get_nodes_with_lowest_degree(graph,2*n_perturbations)

        rows = nodes[:n_perturbations]
        cols = nodes[n_perturbations:]
        print(f'{self.__class__.__name__}: obtained nodes in {time.time()-tick}')

        tick = time.time()
        # dgl_graph = dgl.from_networkx(graph)
        # e0 = [e[0] for e in graph.edges()]
        # e1 = [e[1] for e in graph.edges()]
        # dgl_graph.add_edges(e1,e0)
        # bfs_nodes_generator(dgl_graph,rows[0])
        # print(f'{self.__class__.__name__}: computed SSSP on one node in {time.time()-tick}')
        # tick = time.time()
        # bfs_nodes = {u:bfs_nodes_generator(dgl_graph,u) for u in rows}
        # distance = {u:{v.item():i for i,lvl in enumerate(bfs_nodes[u]) for v in lvl} for u in rows}
        # distance = {u:{v:distance[u][v] if v in distance[u] else self.INF for v in cols} for u in rows}

        distance = bfs(graph, rows) # = {u:nx.single_source_shortest_path_length(graph,u) for u in rows}
        distance = {u:{v:distance[u][v] for v in cols} for u in rows}
        print(f'{self.__class__.__name__}: computed distance in {time.time()-tick}')

        tick = time.time()
        mtx = np.array([np.array(list(distance[u].values())) for u in distance])

        i_u = {i:u for i,u in enumerate(distance)}
        i_v = {i:v for i,v in enumerate(distance[list(distance.keys())[0]])}

        u,v = linear_sum_assignment(-mtx)
        print(f'{self.__class__.__name__}: computed assignment in {time.time()-tick}')

        tick = time.time()
        edges = [[i_u[i],i_v[j]] for i,j in zip(u,v)]
        graph.add_edges_from(edges)
        print(f'{self.__class__.__name__}: added edges in {time.time()-tick}')

        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj
