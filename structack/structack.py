import math
from abc import ABC

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
import community
# from dgl.traversal import bfs_nodes_generator
from structack.bfs import bfs
import structack.node_selection as ns
import structack.node_connection as nc


class StructackBase(BaseAttack):

    def __init__(self):
        super(StructackBase, self).__init__(None, None)
        self.modified_adj = None

    def node_selection(self, graph, n):
        # need to return sorted based on a criterion
        return ns.get_random_nodes(graph, n)

    def node_connection(self, adj, nodes, n_perturbations):
        return nc.random_connection(adj, nodes, n_perturbations)

    def get_perturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)

        # selection
        nodes = self.node_selection(graph, n_perturbations * 2)

        # connection
        edges = self.node_connection(adj, nodes, n_perturbations)

        # apply perturbation
        graph.add_edges_from(edges)
        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj

    def attack(self, ori_adj, n_perturbations):
        self.modified_adj = self.get_perturbed_adj(ori_adj, n_perturbations)


def build_custom(node_selection, node_connection, dataset_name):

    class StructackTemp(StructackBase):

        def node_selection(self, graph, n):
            return node_selection(graph, n, dataset_name=dataset_name)

        def node_connection(self, adj, nodes, n_perturbations):
            return node_connection(adj, nodes, n_perturbations, dataset_name=dataset_name)

    return StructackTemp()

# class StructackCustom(StructackBase):

#     def __init__(self):
#         super(StructackCustom, self).__init__()
#         self.node_selection = selection
#         self.node_connection = connection


class StructackOneEnd(StructackBase):

    def __init__(self, degree_percentile_range=[0, 1]):
        super(StructackOneEnd, self).__init__()
        self.frm, self.to = degree_percentile_range

    def node_selection(self, graph, n):
        node_pool = ns.get_nodes_with_degree_percentile(graph, self.frm, self.to)
        rows = np.random.choice(node_pool, n // 2)
        cols = np.random.choice(graph.nodes(), n // 2)
        return rows + cols

    def node_connection(self, adj, nodes, n_perturbations):
        rows = nodes[:n_perturbations]
        cols = nodes[n_perturbations:]
        return [[u, v] for u, v in zip(rows, cols)]


class StructackBothEnds(StructackBase):

    def __init__(self, degree_percentile_range=[0, 1, 0, 1]):
        super(StructackBothEnds, self).__init__()
        self.frm1, self.to1, self.frm2, self.to2 = degree_percentile_range

    def node_selection(self, graph, n):
        rows = np.random.choice(ns.get_nodes_with_degree_percentile(graph, self.frm1, self.to1), n // 2)
        cols = np.random.choice(ns.get_nodes_with_degree_percentile(graph, self.frm2, self.to2), n // 2)
        return rows + cols

    def node_connection(self, adj, nodes, n_perturbations):
        rows = nodes[:n_perturbations]
        cols = nodes[n_perturbations:]
        return [[u, v] for u, v in zip(rows, cols)]


class StrcutackDegreeBase(StructackBase):

    def node_selection(self, graph, n):
        return ns.get_nodes_with_lowest_degree(graph, n)


class StructackDegreeRandomLinking(StrcutackDegreeBase):

    def __init__(self):
        super(StructackDegreeRandomLinking, self).__init__()
        self.modified_adj = None

    def node_connection(self, adj, nodes, n_perturbations):
        return nc.random_connection(adj, nodes, n_perturbations)


class StructackDegree(StrcutackDegreeBase):

    def __init__(self):
        super(StructackDegree, self).__init__()
        self.modified_adj = None

    def node_connection(self, adj, nodes, n_perturbations):
        return nc.sorted_connection(adj, nodes, n_perturbations)


class StructackEigenvectorCentrality(StrcutackDegreeBase):

    def __init__(self):
        super(StructackEigenvectorCentrality, self).__init__()
        self.modified_adj = None

    def node_selection(self, graph, n):
        return ns.get_nodes_with_lowest_eigenvector_centrality(graph, n)


class StructackBetweennessCentrality(StrcutackDegreeBase):

    def __init__(self):
        super(StructackBetweennessCentrality, self).__init__()
        self.modified_adj = None

    def node_selection(self, graph, n):
        return ns.get_nodes_with_lowest_betweenness_centrality(graph, n)


class StructackClosenessCentrality(StructackBase):
    def __init__(self):
        super(StructackClosenessCentrality, self).__init__()
        self.modified_adj = None

    def node_selection(self, graph, n):
        return ns.get_nodes_with_lowest_closeness_centrality(graph, n)


class StructackPageRank(StructackBase):
    def __init__(self):
        super(StructackPageRank, self).__init__()
        self.modified_adj = None

    def node_selection(self, graph, n):
        return ns.get_nodes_with_lowest_pagerank(graph, n)


class StructackKatzSimilarity(StructackBase):
    def __init__(self):
        super(StructackKatzSimilarity, self).__init__()
        self.modified_adj = None

    def node_connection(self, adj, nodes, n_perturbations):
        return nc.katz_connection(adj, nodes, n_perturbations)



class StructackCommunity(StructackBase):
    def __init__(self):
        super(StructackCommunity, self).__init__()
        self.modified_adj = None

    def node_connection(self, adj, nodes, n_perturbations):
        return nc.community_connection(adj, nodes, n_perturbations)


class StructackDistance(StructackBase):
    def __init__(self):
        super(StructackDistance, self).__init__()
        self.modified_adj = None

    def node_connection(self, adj, nodes, n_perturbations):
        return nc.distance_hungarian_connection(adj, nodes, n_perturbations)

class StructackDegreeDistance(StructackBase):
    def __init__(self):
        super(StructackDegreeDistance, self).__init__()
        self.INF = 1e9 + 7
        self.modified_adj = None

    def node_selection(self, graph, n):
        return ns.get_nodes_with_lowest_degree(graph, n)

    def node_connection(self, adj, nodes, n_perturbations):
        return nc.distance_hungarian_connection(adj, nodes, n_perturbations)


    # def get_perturbed_adj(self, adj, n_perturbations):
    #     graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph)
    #     n = adj.shape[0]
    #     tick = time.time()
    #     # select nodes
    #     nodes = self.get_nodes_with_lowest_degree(graph, 2 * n_perturbations)

    #     rows = nodes[:n_perturbations]
    #     cols = nodes[n_perturbations:]
    #     print(f'{self.__class__.__name__}: obtained nodes in {time.time() - tick}')

    #     tick = time.time()
    #     # dgl_graph = dgl.from_networkx(graph)
    #     # e0 = [e[0] for e in graph.edges()]
    #     # e1 = [e[1] for e in graph.edges()]
    #     # dgl_graph.add_edges(e1,e0)
    #     # bfs_nodes_generator(dgl_graph,rows[0])
    #     # print(f'{self.__class__.__name__}: computed SSSP on one node in {time.time()-tick}')
    #     # tick = time.time()
    #     # bfs_nodes = {u:bfs_nodes_generator(dgl_graph,u) for u in rows}
    #     # distance = {u:{v.item():i for i,lvl in enumerate(bfs_nodes[u]) for v in lvl} for u in rows}
    #     # distance = {u:{v:distance[u][v] if v in distance[u] else self.INF for v in cols} for u in rows}

    #     distance = bfs(graph, rows)  # = {u:nx.single_source_shortest_path_length(graph,u) for u in rows}
    #     distance = {u: {v: distance[u][v] for v in cols} for u in rows}
    #     print(f'{self.__class__.__name__}: computed distance in {time.time() - tick}')

    #     tick = time.time()
    #     mtx = np.array([np.array(list(distance[u].values())) for u in distance])

    #     i_u = {i: u for i, u in enumerate(distance)}
    #     i_v = {i: v for i, v in enumerate(distance[list(distance.keys())[0]])}

    #     u, v = linear_sum_assignment(-mtx)
    #     print(f'{self.__class__.__name__}: computed assignment in {time.time() - tick}')

    #     tick = time.time()
    #     edges = [[i_u[i], i_v[j]] for i, j in zip(u, v)]
    #     graph.add_edges_from(edges)
    #     print(f'{self.__class__.__name__}: added edges in {time.time() - tick}')

    #     modified_adj = nx.to_scipy_sparse_matrix(graph)
    #     return modified_adj


class StructackRangeDistance(StructackBase):
    def __init__(self, distance_percentile_range=[0, 1]):
        super(StructackRangeDistance, self).__init__()
        self.frm, self.to = distance_percentile_range
        self.modified_adj = None

    def get_perturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph)
        n = adj.shape[0]
        tick = time.time()
        # select nodes
        rows = np.random.choice(graph.nodes(), size=n_perturbations, replace=(len(graph.nodes()) < 2 * n_perturbations))

        print(f'{self.__class__.__name__}: obtained nodes in {time.time() - tick}')

        tick = time.time()
        distance = {u: nx.single_source_shortest_path_length(graph, u) for u in rows}

        frm = int(self.frm * n)
        to = int(self.to * n)
        distance = {u: sorted(distance[u].items(), key=lambda x: x[1])[frm:to] for u in distance}
        idx = {u: np.random.choice(len(distance[u])) for u in distance}
        print(f'{self.__class__.__name__}: computed distance in {time.time() - tick}')

        tick = time.time()
        edges = [[u, distance[u][idx[u]][0]] for u in distance]
        distance = [distance[u][idx[u]][1] for u in distance]
        # print(distance)
        # print(edges)
        # exit(0)
        graph.add_edges_from(edges)
        print(f'{self.__class__.__name__}: added edges in {time.time() - tick}')

        # print(distance)
        # print(edges)
        # exit(0)
        self.mean_distance = np.mean(distance)
        print(f'mean distance = {self.mean_distance:.2f}')

        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj

