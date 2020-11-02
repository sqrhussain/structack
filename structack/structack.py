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
import community
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
        if len(nodes) < n: # repeat the list until it's longer than n
            nodes = nodes * int(n/len(nodes) + 1)
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
    

class StructackEigenvectorCentrality(StructackBase):
    def __init__(self):
        super(StructackEigenvectorCentrality, self).__init__()
        self.modified_adj = None
    def get_purturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
        n = adj.shape[0]

        # select nodes
        nodes = self.get_nodes_with_lowest_eigenvector_centrality(graph,2*n_perturbations)

        # perturb
        rows = nodes[:n_perturbations]
        cols = nodes[n_perturbations:]
        edges = [[u,v] for u,v in zip(rows, cols)]
        graph.add_edges_from(edges)

        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj
    def get_nodes_with_lowest_eigenvector_centrality(self, G, n):
        nodes = sorted(nx.eigenvector_centrality(G).items(), key=lambda x: x[1])
        if len(nodes) < n: # repeat the list until it's longer than n
            nodes = nodes * int(n/len(nodes) + 1)
        nodes = nodes[:n]
        return [x[0] for x in nodes]
    

class StructackBetweennessCentrality(StructackBase):
    def __init__(self):
        super(StructackBetweennessCentrality, self).__init__()
        self.modified_adj = None
    def get_purturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
        n = adj.shape[0]

        # select nodes
        nodes = self.get_nodes_with_lowest_betweenness_centrality(graph,2*n_perturbations)

        # perturb
        rows = nodes[:n_perturbations]
        cols = nodes[n_perturbations:]
        edges = [[u,v] for u,v in zip(rows, cols)]
        graph.add_edges_from(edges)

        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj
    def get_nodes_with_lowest_betweenness_centrality(self, G, n):
        nodes = sorted(nx.betweenness_centrality(G).items(), key=lambda x: x[1])
        if len(nodes) < n: # repeat the list until it's longer than n
            nodes = nodes * int(n/len(nodes) + 1)
        nodes = nodes[:n]
        return [x[0] for x in nodes]
    
    
class StructackClosenessCentrality(StructackBase):
    def __init__(self):
        super(StructackClosenessCentrality, self).__init__()
        self.modified_adj = None
    def get_purturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
        n = adj.shape[0]

        # select nodes
        nodes = self.get_nodes_with_lowest_closeness_centrality(graph,2*n_perturbations)

        # perturb
        rows = nodes[:n_perturbations]
        cols = nodes[n_perturbations:]
        edges = [[u,v] for u,v in zip(rows, cols)]
        graph.add_edges_from(edges)

        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj
    def get_nodes_with_lowest_closeness_centrality(self, G, n):
        nodes = sorted(nx.closeness_centrality(G).items(), key=lambda x: x[1])
        if len(nodes) < n: # repeat the list until it's longer than n
            nodes = nodes * int(n/len(nodes) + 1)
        nodes = nodes[:n]
        return [x[0] for x in nodes]
    
      
class StructackPageRank(StructackBase):
    def __init__(self):
        super(StructackPageRank, self).__init__()
        self.modified_adj = None
    def get_purturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
        n = adj.shape[0]

        # select nodes
        nodes = self.get_nodes_with_lowest_pagerank(graph,2*n_perturbations)

        # perturb
        rows = nodes[:n_perturbations]
        cols = nodes[n_perturbations:]
        edges = [[u,v] for u,v in zip(rows, cols)]
        graph.add_edges_from(edges)

        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj
    def get_nodes_with_lowest_pagerank(self, G, n):
        nodes = sorted(nx.pagerank(G).items(), key=lambda x: x[1])
        if len(nodes) < n: # repeat the list until it's longer than n
            nodes = nodes * int(n/len(nodes) + 1)
        nodes = nodes[:n]
        return [x[0] for x in nodes]
    
class StructackKatzSimilarity(StructackBase):
    def __init__(self):
        super(StructackKatzSimilarity, self).__init__()
        self.modified_adj = None
    def get_purturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
        n = adj.shape[0]
        tick = time.time()

        # select nodes
        nodes = np.random.choice(graph.nodes(), size=2*n_perturbations, replace=(len(graph.nodes())<2*n_perturbations))

        # perturb
        rows = nodes[:n_perturbations]
        cols = self.get_nodes_with_lowest_katz_similarity(rows, adj, graph)
        
        edges = [[u,v] for u,v in zip(rows, cols)]
        graph.add_edges_from(edges)

        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj
    def get_nodes_with_lowest_katz_similarity(self, rows, adj, graph):
        alpha = 0.1
        n_steps = 4
        temp_sum = 0
        for i in range(1, n_steps):
            temp_sum += alpha**i * adj**i
        D_sqrt = nx.linalg.laplacianmatrix.laplacian_matrix(graph) + adj
        D_sqrt.data = np.sqrt(D_sqrt.data)
        sigma = D_sqrt*(temp_sum + sp.identity(adj.shape[0], format='csr'))*D_sqrt
        cols = []
        for i in rows:
            cols.append(np.sort(sigma[i,:].nonzero()[1])[sigma[sigma[i,:].nonzero()[0], np.sort(sigma[i,:].nonzero()[1])].argmin()])
        return cols
    
class StructackCommunity(StructackBase):
    def __init__(self):
        super(StructackCommunity, self).__init__()
        self.modified_adj = None
    def get_purturbed_adj(self, adj, n_perturbations):
        graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
        n = adj.shape[0]
        tick = time.time()

        # select nodes
        nodes = np.random.choice(graph.nodes(), size=2*n_perturbations, replace=(len(graph.nodes())<2*n_perturbations))

        # perturb
        rows = nodes[:n_perturbations]
        cols = self.get_nodes_with_least_intercommunity_edges(rows, graph)
        
        edges = [[u,v] for u,v in zip(rows, cols)]
        graph.add_edges_from(edges)

        modified_adj = nx.to_scipy_sparse_matrix(graph)
        return modified_adj
    def get_nodes_with_least_intercommunity_edges(self, rows, graph):
        node_community_mapping = community.community_louvain.best_partition(graph) 
        community_node_mapping = {}
        community_edge_counts = {}

        for edge in graph.edges:
            if node_community_mapping[edge[0]] not in community_node_mapping:
                community_node_mapping[node_community_mapping[edge[0]]] = []
            if node_community_mapping[edge[1]] not in community_node_mapping:
                community_node_mapping[node_community_mapping[edge[1]]] = []
            if edge[0] not in community_node_mapping[node_community_mapping[edge[0]]]:
                community_node_mapping[node_community_mapping[edge[0]]].append(edge[0])
            if edge[1] not in community_node_mapping[node_community_mapping[edge[1]]]:
                community_node_mapping[node_community_mapping[edge[1]]].append(edge[1])
            if node_community_mapping[edge[0]] == node_community_mapping[edge[1]]:
                continue
            if (node_community_mapping[edge[0]], node_community_mapping[edge[1]]) not in community_edge_counts:
                community_edge_counts[(node_community_mapping[edge[0]], node_community_mapping[edge[1]])] = 0
            if (node_community_mapping[edge[1]], node_community_mapping[edge[0]]) not in community_edge_counts:
                community_edge_counts[(node_community_mapping[edge[1]], node_community_mapping[edge[0]])] = 0
            community_edge_counts[(node_community_mapping[edge[0]], node_community_mapping[edge[1]])] += 1
            community_edge_counts[(node_community_mapping[edge[1]], node_community_mapping[edge[0]])] += 1
    
        adj_community_rows = []
        adj_community_cols = []
        adj_community_data = []

        for key in community_edge_counts:
            adj_community_rows.append(key[0])
            adj_community_cols.append(key[1])
            adj_community_data.append(community_edge_counts[key])
    
        adj_community = sp.csr_matrix((adj_community_data, (adj_community_rows, adj_community_cols)))
        adj_community = adj_community.toarray().astype('float')
        adj_community[adj_community == 0] = np.nan

        distant_communities = {}
        for community_id in range(adj_community.shape[0]):
            distant_community_id = np.argpartition(adj_community[community_id], 1)[:1][0]
            while distant_community_id == 0 or distant_community_id == community_id:
                distant_community_id = np.random.choice(adj_community.shape[0], 1)[0]
            distant_communities[community_id] = distant_community_id

        cols = []
        for node_id in rows:
            community_id = node_community_mapping[node_id]
            distant_community_id = distant_communities[community_id]
            cols.append(np.random.choice(community_node_mapping[distant_community_id],1)[0])
        return cols

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
    


