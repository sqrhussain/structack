import numpy as np
import time
import networkx as nx
from scipy.optimize import linear_sum_assignment
import time
import community
from structack.bfs import bfs
import scipy.sparse as sp
import scipy.sparse.linalg as spalg
import pickle
import os
import pickle


def sorted_connection(adj, nodes, n_perturbations, dataset_name=None):
    rows = nodes[:n_perturbations]
    cols = nodes[n_perturbations:]
    return [[u, v] for u, v in zip(rows, cols)]


def random_connection(adj, nodes, n_perturbations, dataset_name=None):
    np.random.shuffle(nodes)
    rows = nodes[:n_perturbations]
    cols = nodes[n_perturbations:]
    return [[u, v] for u, v in zip(rows, cols)]


def distance_hungarian_connection(adj, nodes, n_perturbations, dataset_name=None):
    graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
    rows = nodes[:n_perturbations]
    cols = nodes[n_perturbations:]
    precomputed_path = f'data/tmp/{dataset_name}_distance.pkl'
    if dataset_name is not None and os.path.exists(precomputed_path):
        print("Loading precomputed_distance...")
        with open(precomputed_path,'rb') as ff:
            precomputed_distance = pickle.load(ff)
    else:
        precomputed_distance = {}

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

    new_rows = [u for u in rows if u not in precomputed_distance]
    print(f'{len(new_rows)} distances to be computed' )
    new_distances = bfs(graph, new_rows)
    precomputed_distance = {**precomputed_distance, **new_distances}
    distance = {u: {v: precomputed_distance[u][v] for v in cols} for u in rows}
    print(f'distance_connection: computed distance in {time.time() - tick}')

    tick = time.time()
    mtx = np.array([np.array(list(distance[u].values())) for u in distance])

    i_u = {i: u for i, u in enumerate(distance)}
    i_v = {i: v for i, v in enumerate(distance[list(distance.keys())[0]])}

    u, v = linear_sum_assignment(-mtx)
    print(f'distance_connection: computed assignment in {time.time() - tick}')

    tick = time.time()
    if dataset_name is not None:
        with open(precomputed_path,'wb') as ff:
            precomputed_distance = pickle.dump(precomputed_distance, ff)
    return [[i_u[i], i_v[j]] for i, j in zip(u, v)]


def katz_hungarian_connection(adj, nodes, n_perturbations, threshold=0.000001, nsteps=10000, dataset_name=None):
    graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
    rows = nodes[:n_perturbations]
    cols = nodes[n_perturbations:]
    precomputed_path = f'data/tmp/{dataset_name}_katz.pkl'
    if dataset_name is not None and os.path.exists(precomputed_path):
        print("Loading precomputed_katz...")
        with open(precomputed_path,'r') as ff:
            sigma = pickle.load(open(precomputed_path, 'rb'))
    else:
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
        sigma = sigma.toarray().astype('float')
        if dataset_name is not None:
            pickle.dump(sigma, open(precomputed_path, "wb" ) )

    similarity = {u: {v: sigma[u][v] for v in cols} for u in rows}

    mtx = np.array([np.array(list(similarity[u].values())) for u in similarity])

    i_u = {i: u for i, u in enumerate(similarity)}
    i_v = {i: v for i, v in enumerate(similarity[list(similarity.keys())[0]])}

    u, v = linear_sum_assignment(+mtx)

    return [[i_u[i], i_v[j]] for i, j in zip(u, v)] 
    # cols = []
    # for i in rows:
    #     cols.append(np.argmin(sigma[i, :].todense(), axis=1)[0,0])
    # return [[u, v] for u, v in zip(rows, cols)]


def community_hungarian_connection(adj, nodes, n_perturbations, dataset_name=None):
    graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
    rows = nodes[:n_perturbations]
    cols = nodes[n_perturbations:]

    precomputed_path = f'data/tmp/{dataset_name}_communities.pkl'
    if dataset_name is not None and os.path.exists(precomputed_path):
        print("Loading precomputed_communities...")
        with open(precomputed_path,'r') as ff:
            node_community_mapping = pickle.load(open(precomputed_path, 'rb'))
    else:
        node_community_mapping = community.community_louvain.best_partition(graph)
        if dataset_name is not None:
            pickle.dump(node_community_mapping, open(precomputed_path, "wb" ))
    
    community_node_mapping = {}
    community_edge_counts = {}
    communities = list(set(node_community_mapping.values()))
    for source_community in communities:
        for target_community in communities:
            community_edge_counts[(source_community, target_community)] = 0
            community_edge_counts[(target_community, source_community)] = 0

    for edge in graph.edges:
        if node_community_mapping[edge[0]] not in community_node_mapping:
            community_node_mapping[node_community_mapping[edge[0]]] = []
        if node_community_mapping[edge[1]] not in community_node_mapping:
            community_node_mapping[node_community_mapping[edge[1]]] = []
        if edge[0] not in community_node_mapping[node_community_mapping[edge[0]]]:
            community_node_mapping[node_community_mapping[edge[0]]].append(edge[0])
        if edge[1] not in community_node_mapping[node_community_mapping[edge[1]]]:
            community_node_mapping[node_community_mapping[edge[1]]].append(edge[1])
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

    similarity = {u: {v: adj_community[node_community_mapping[u]][node_community_mapping[v]] for v in cols} for u in rows}

    mtx = np.array([np.array(list(similarity[u].values())) for u in similarity])

    i_u = {i: u for i, u in enumerate(similarity)}
    i_v = {i: v for i, v in enumerate(similarity[list(similarity.keys())[0]])}

    u, v = linear_sum_assignment(+mtx)

    return [[i_u[i], i_v[j]] for i, j in zip(u, v)]

    # distant_communities = {}
    # for community_id in range(adj_community.shape[0]):
    #     distant_community_id = np.argwhere(adj_community[community_id,:]==np.min(adj_community[community_id,:])).flatten()
    #     distant_communities[community_id] = distant_community_id

    # cols = []
    # for node_id in rows:
    #     community_id = node_community_mapping[node_id]
    #     distant_community_id = distant_communities[community_id]
    #     cols.append(np.random.choice(community_node_mapping[distant_community_id], 1)[0])
    # return [[u, v] for u, v in zip(rows, cols)]
