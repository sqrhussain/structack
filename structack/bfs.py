import subprocess
import networkx as nx
import os
gt = True
try:
    from graph_tool.all import *
    from graph_tool.search import bfs_search, BFSVisitor
    class SimpleVisitor(BFSVisitor):

        def __init__(self, dist):
            # Not initilizing dist[source]?
            self.dist = dist

        def tree_edge(self, e):
            self.dist[e.target()] = self.dist[e.source()] + 1
except:
    gt = False
    print('graph-tool is not installed, using networkx instead.')


bin_path = './exec/bfs'
def bfs_one_source_cpp(graph, source):
    # build input string
    nmk = f'{len(graph.nodes())} {len(graph.edges())} 1\n'
    src = str(source) + '\n'
    edg = '\n'.join(str(x[0]) + ' ' + str(x[1]) for x in graph.edges()) +'\n'
    output = subprocess.run([bin_path],capture_output=True,input=nmk+src+edg,text=True).stdout
    return eval(output[source])



def graph_from_nx(graph):
    g = Graph(directed=False)
    g.add_edge_list(list(graph.edges()))
    return g


def bfs_one_source(graph, source):
    g = graph_from_nx(graph)
    dist = g.new_vertex_property("int")
    bfs_search(g, g.vertex(source), SimpleVisitor(dist))
    print(dist.a)
    exit(1)
    return dist.a


def bfs(graph, sources):
    gt = False
    if not gt:
        return {u:nx.single_source_shortest_path_length(graph,u) for u in sources}
    

    # if not os.path.exists(bin_path):
    #     print("WARN: Can't find C++ implementation. Using networkx." )
    #     print(f"INFO: To find the shortest path faster, please halt the process, run 'g++ -std=c++11 -o {bin_path}', and then run the experiment again.")
    #     return {u:nx.single_source_shortest_path_length(graph,u) for u in sources}

    # build input string
    # nmk = f'{len(graph.nodes())} {len(graph.edges())} {len(sources)}\n'
    # src = ' '.join(str(x) for x in sources) + '\n'
    # edg = '\n'.join(str(x[0]) + ' ' + str(x[1]) for x in graph.edges()) +'\n'

    # execute the process
    # output = subprocess.run([bin_path],capture_output=True,input=nmk+src+edg,text=True).stdout

    # return eval(output)

    return {u:bfs_one_source(graph, u) for u in sources}

