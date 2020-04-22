import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
import matplotlib.pyplot as plt
import sys
import random


def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """

    # TODO: your code here!
    start_with = 1

    all_nodes = set(G)
    if start_with is None:
        start_with = arbitrary_element(all_nodes)
    if start_with not in G:
        raise nx.NetworkXError('node {} is not in G'.format(start_with))
    dominating_set = {start_with}
    dominated_nodes = set(G[start_with])
    remaining_nodes = all_nodes - dominated_nodes - dominating_set
    while remaining_nodes:
        # Choose an arbitrary node and determine its undominated neighbors.
        v = remaining_nodes.pop()
        undominated_neighbors = set(G[v]) - dominating_set
        # Add the node to the dominating set and the neighbors to the
        # dominated set. Finally, remove all of those nodes from the set
        # of remaining nodes.
        dominating_set.add(v)
        dominated_nodes |= undominated_neighbors
        remaining_nodes -= undominated_neighbors
    T = G.subgraph(dominating_set)
    print(T.nodes())
    print(T.edges())
    print(nx.is_tree(T))
    print(nx.is_dominating_set(G, T.nodes))
    return nx.minimum_spanning_tree(G.subgraph(dominating_set))

def potential_fct(black, gray, white, G):
    # returns number of white vertices + number of black components
    # black components = number of connected components induced by black vertices
    num_black_components = nx.number_connected_components(G.subgraph(black))
    return len(white) + num_black_components

def node_potential(black, gray, white, node):
    # color it black
    # color adjacent white vertices gray
    # call potential function
    white.remove(node)
    black.add(node)
    for adj_node in G[node]:
        if adj_node in white:
            white.remove(adj_node)
            gray.add(adj_node)
    return potential_fct(black, gray, white, G)

def greedy(G):
    black = set() # vertices in T
    gray = set() # vertices not in T but adjacent to T
    white = set(G) # vertices not in C and not adjacent to T

    flag = 1
    while flag == 1:
        # if there exists a white or gray vertex such that
        # coloring it in black and its adjacent white vertices in gray
        # would reduce the value of potential function
        initial_potential = potential_fct(black,gray,white,G)
        potentials = {}
        for node in white:
            b = black.copy()
            g = gray.copy()
            w = white.copy()
            potentials[node] = initial_potential - node_potential(b, g, w, node)
        for node in gray:
            b = black.copy()
            g = gray.copy()
            w = white.copy()
            potentials[node] = initial_potential - node_potential(b, g, w, node)
        print(potentials)
        if len(potentials) == 0:
            break
        if max(potentials.values()) > 1:
            chosen_node = max(potentials,key=potentials.get)
            print("chosen node:",chosen_node)
            black.add(chosen_node)
            if chosen_node in white:
                white.remove(chosen_node)
            if chosen_node in gray:
                gray.remove(chosen_node)
        else:
            flag = 0

    return nx.minimum_spanning_tree(G.subgraph(black))

def spanning_edges(G, weight='weight', data=True):
    from networkx.utils import UnionFind
    subtrees = UnionFind()
    #edges = sorted(G.edges(data=True), key=lambda t:t[2].get(weight, 1)+random.random()*20)
    edges = sorted(G.edges(data=True), key=lambda t: t[2].get(weight, 1))
    for u, v, d in edges:
        if subtrees[u] != subtrees[v]:
            if data:
                yield (u, v, d)
            else:
                yield (u, v)
            subtrees.union(u, v)

def spanning_tree(G):
    min_spanning_edges = nx.minimum_spanning_edges(G, algorithm='prim', weight='weight', data=True)
    T = nx.Graph(min_spanning_edges)
    # Add isolated nodes
    if len(T) != len(G):
        T.add_nodes_from([n for n, d in G.degree().items() if d == 0])
    # Add node and graph attributes as shallow copy
    # for n in T:
    #     T.node[n] = G.node[n].copy()
    # T.graph = G.graph.copy()
    return T

def leaf_pruning(T):
    nodes_to_remove = []
    for node in T.nodes():
        if T.degree(node)==1:
            nodes_to_remove += [node]
    for node in nodes_to_remove:
        T.remove_node(node)
    return T

def shortest_paths(G,source):
    paths = nx.single_source_dijkstra_path(G,source)
    edges = []
    for path in paths.values():
        #print(path)
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            edges += [(u,v,G.get_edge_data(u,v))]
    #print(edges)
    T = nx.Graph(edges)
    return T

def draw(G,T):
    pos = nx.spring_layout(G)
    nx.draw_networkx(G,pos=pos, node_color='blue')
    nx.draw_networkx(T, pos=pos, node_color='red', edge_color='red')
    plt.show()

# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    #assert len(sys.argv) == 2
    start = 1
    end = 400
    size = 'large'
    for i in range(start,1+end):
        path = 'inputs/' + size + '-'+str(i)+'.in'
        G = read_input_file(path)

        if G.number_of_nodes() <= 2:
            T = G.subgraph(0)
            write_output_file(T, 'outputs/' + size + '-' + str(i) + '.out')
            print(str(i), "is done.")
            continue

        T = leaf_pruning(shortest_paths(G,0))
        min_dist = average_pairwise_distance_fast(T)
        for j in range(1,G.number_of_nodes()-1):
            new_T = leaf_pruning(shortest_paths(G, j))
            dist = average_pairwise_distance_fast(new_T)
            if dist < min_dist:
                min_dist = dist
                T = new_T

        assert is_valid_network(G, T)
        #draw(G,T)
        print("Average  pairwise distance: {}".format(average_pairwise_distance_fast(T)))
        print(str(i),"is done.")
        write_output_file(T, 'outputs/' + size + '-'+str(i)+'.out')
