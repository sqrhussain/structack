import numpy as np
import time
import networkx as nx
from scipy.optimize import linear_sum_assignment
import time
import community
from structack.bfs import bfs
import scipy.sparse as sp
import scipy.sparse.linalg as spalg


def sorted_connection(adj, nodes, n_perturbations):
    rows = nodes[:n_perturbations]
    cols = nodes[n_perturbations:]
    return [[u, v] for u, v in zip(rows, cols)]


def random_connection(adj, nodes, n_perturbations):
    np.random.shuffle(nodes)
    rows = nodes[:n_perturbations]
    cols = nodes[n_perturbations:]
    return [[u, v] for u, v in zip(rows, cols)]


def distance_hungarian_connection(adj, nodes, n_perturbations):
    graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
    rows = nodes[:n_perturbations]
    cols = nodes[n_perturbations:]

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

    distance = bfs(graph, rows)  # = {u:nx.single_source_shortest_path_length(graph,u) for u in rows}
    distance = {u: {v: distance[u][v] for v in cols} for u in rows}
    print(f'distance_connection: computed distance in {time.time() - tick}')

    tick = time.time()
    mtx = np.array([np.array(list(distance[u].values())) for u in distance])

    i_u = {i: u for i, u in enumerate(distance)}
    i_v = {i: v for i, v in enumerate(distance[list(distance.keys())[0]])}

    u, v = linear_sum_assignment(-mtx)
    print(f'distance_connection: computed assignment in {time.time() - tick}')

    tick = time.time()
    return [[i_u[i], i_v[j]] for i, j in zip(u, v)]


def katz_connection(adj, nodes, n_perturbations, threshold=0.000001, nsteps=100):
    graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
    rows = nodes[:n_perturbations]
    D = nx.linalg.laplacianmatrix.laplacian_matrix(graph) + adj
    D_inv = spalg.inv(D)
    D_invA = D_inv * adj
    l,v = spalg.eigs(D_invA, k=1, which="LR")
    lmax = l[0].real
    alpha = (1/lmax) * 0.9
    sigma = sp.csr_matrix(D_invA.shape, dtype=np.float)
    print('Calculate sigma matrix')
    for i in range(nsteps):
        sigma_new = alpha *D_invA*sigma + sp.identity(adj.shape[0], dtype=np.float, format='csr')
        diff = abs(spalg.norm(sigma, 1) - spalg.norm(sigma_new, 1))
        sigma = sigma_new
        print(diff)
        if diff < threshold:
            break
        print('Number of steps taken: ' + str(i))
    cols = []
    for i in rows:
        cols.append(np.sort(sigma[i, :].nonzero()[1])[
                        sigma[sigma[i, :].nonzero()[0], np.sort(sigma[i, :].nonzero()[1])].argmin()])
    return [[u, v] for u, v in zip(rows, cols)]    
    

def community_connection(adj, nodes, n_perturbations):
    graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
    rows = nodes[:n_perturbations]

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
        cols.append(np.random.choice(community_node_mapping[distant_community_id], 1)[0])
    return [[u, v] for u, v in zip(rows, cols)]
