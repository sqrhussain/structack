import networkx as nx
import numpy as np
import os
import pickle


def get_random_nodes(graph, n, dataset_name=None):
    return np.random.choice(graph.nodes(), size=n,
                            replace=(len(graph.nodes()) < n))


def get_nodes_with_degree_percentile(graph, frm, to, dataset_name=None):
    nodes = sorted(graph.degree, key=lambda x: x[1])
    length = len(nodes)
    frm = int(length * frm)
    to = int(length * to + 1)
    nodes = nodes[frm:to]
    return [x[0] for x in nodes]


def get_nodes_with_lowest_degree(graph, n, dataset_name=None):
    nodes = sorted(graph.degree, key=lambda x: x[1])
    if len(nodes) < n:  # repeat the list until it's longer than n
        nodes *= int(n / len(nodes) + 1)
    nodes = nodes[:n]
    return [x[0] for x in nodes]


def get_nodes_with_lowest_eigenvector_centrality(graph, n, dataset_name=None):
    precomputed_path = f'data/tmp/{dataset_name}_eigenvector_centralities.pkl'
    if dataset_name is not None and os.path.exists(precomputed_path):
        print("Loading precomputed_eigenvector_centralities...")
        with open(precomputed_path,'r') as ff:
            eigenvector_centralities = pickle.load(open(precomputed_path, 'rb'))
    else:
        eigenvector_centralities = nx.eigenvector_centrality(graph)
        if dataset_name is not None:
            pickle.dump(eigenvector_centralities, open(precomputed_path, "wb" ) )
        
    nodes = sorted(eigenvector_centralities.items(), key=lambda x: x[1])
    if len(nodes) < n:  # repeat the list until it's longer than n
        nodes *= int(n / len(nodes) + 1)
    nodes = nodes[:n]
    return [x[0] for x in nodes]


def get_nodes_with_lowest_betweenness_centrality(graph, n, dataset_name=None):
    precomputed_path = f'data/tmp/{dataset_name}_betweenness_centralities.pkl'
    if dataset_name is not None and os.path.exists(precomputed_path):
        print("Loading precomputed_betweenness_centralities...")
        with open(precomputed_path,'r') as ff:
            betweenness_centralities = pickle.load(open(precomputed_path, 'rb'))
    else:
        betweenness_centralities = nx.betweenness_centrality(graph)
        if dataset_name is not None:
            pickle.dump(betweenness_centralities, open(precomputed_path, "wb" ) )
        
    nodes = sorted(betweenness_centralities.items(), key=lambda x: x[1])
    if len(nodes) < n:  # repeat the list until it's longer than n
        nodes = nodes * int(n / len(nodes) + 1)
    nodes = nodes[:n]
    return [x[0] for x in nodes]


def get_nodes_with_lowest_closeness_centrality(graph, n, dataset_name=None):
    precomputed_path = f'data/tmp/{dataset_name}_closeness_centralities.pkl'
    if dataset_name is not None and os.path.exists(precomputed_path):
        print("Loading precomputed_closeness_centralities...")
        with open(precomputed_path,'r') as ff:
            closeness_centralities = pickle.load(open(precomputed_path, 'rb'))
    else:
        closeness_centralities = nx.closeness_centrality(graph)
        if dataset_name is not None:
            pickle.dump(closeness_centralities, open(precomputed_path, "wb" ) )
        
    nodes = sorted(closeness_centralities.items(), key=lambda x: x[1])
    if len(nodes) < n:  # repeat the list until it's longer than n
        nodes *= int(n / len(nodes) + 1)
    nodes = nodes[:n]
    return [x[0] for x in nodes]


def get_nodes_with_lowest_pagerank(graph, n, dataset_name=None):
    precomputed_path = f'data/tmp/{dataset_name}_pageranks.pkl'
    if dataset_name is not None and os.path.exists(precomputed_path):
        print("Loading precomputed_pageranks...")
        with open(precomputed_path,'r') as ff:
            pageranks = pickle.load(open(precomputed_path, 'rb'))
    else:
        pageranks = nx.pagerank(graph)
        if dataset_name is not None:
            pickle.dump(pageranks, open(precomputed_path, "wb" ) )
        
    nodes = sorted(pageranks.items(), key=lambda x: x[1])
    if len(nodes) < n:  # repeat the list until it's longer than n
        nodes *= int(n / len(nodes) + 1)
    nodes = nodes[:n]
    return [x[0] for x in nodes]
