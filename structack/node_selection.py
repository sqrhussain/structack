import networkx as nx
import numpy as np


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
    nodes = sorted(nx.eigenvector_centrality(graph).items(), key=lambda x: x[1])
    if len(nodes) < n:  # repeat the list until it's longer than n
        nodes *= int(n / len(nodes) + 1)
    nodes = nodes[:n]
    return [x[0] for x in nodes]


def get_nodes_with_lowest_betweenness_centrality(graph, n, dataset_name=None):
    nodes = sorted(nx.betweenness_centrality(graph).items(), key=lambda x: x[1])
    if len(nodes) < n:  # repeat the list until it's longer than n
        nodes = nodes * int(n / len(nodes) + 1)
    nodes = nodes[:n]
    return [x[0] for x in nodes]


def get_nodes_with_lowest_closeness_centrality(graph, n, dataset_name=None):
    nodes = sorted(nx.closeness_centrality(graph).items(), key=lambda x: x[1])
    if len(nodes) < n:  # repeat the list until it's longer than n
        nodes *= int(n / len(nodes) + 1)
    nodes = nodes[:n]
    return [x[0] for x in nodes]


def get_nodes_with_lowest_pagerank(graph, n, dataset_name=None):
    nodes = sorted(nx.pagerank(graph).items(), key=lambda x: x[1])
    if len(nodes) < n:  # repeat the list until it's longer than n
        nodes *= int(n / len(nodes) + 1)
    nodes = nodes[:n]
    return [x[0] for x in nodes]
