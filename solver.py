import networkx as nx
import math
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
from networkx.algorithms import approximation
import matplotlib.pyplot as plt
import sys
import random

#finds spanning edges
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

#returns spanning tree using Prim's
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

#returns T with leaves pruned
def leaf_pruning(T):
    nodes_to_remove = []
    for node in T.nodes():
        if T.degree(node)==1:
            nodes_to_remove += [node]
    for node in nodes_to_remove:
        T.remove_node(node)
    return T

#returns T with nodes randomly removed
def random_pruning(T):
    for t in T.nodes():
        t_new = T.copy()
        t_new.remove_node(t)
        if is_valid_network(G, t_new):
            T = t_new
    return T

#random as fuck pruning
def more_random_pruning(T):
    pruning = 5*T.number_of_nodes() #how much do u wanna prune
    for i in range(pruning):
        random_node = random.choice(list(T.nodes()))
        t_new = T.copy()
        t_new.remove_node(random_node)
        if t_new.number_of_nodes() > 0 and is_valid_network(G, t_new):
            T = t_new
    return T

#only prune if it improves the avg distance
def dont_always_prune(T):
    min_dist = average_pairwise_distance_fast(T)
    for node in G.nodes():
        if node not in T.nodes():
            t_new = T.copy()
            t_new_nodes = list(t_new.nodes()) + [node] #add the node to the tree
            t_new = G.subgraph(t_new_nodes) # get subgraph of tnew
            for source in t_new_nodes:
                new_T = shortest_paths(t_new, source) # see if shortest path tree has better dist
                dist = average_pairwise_distance_fast(new_T)
                if dist < min_dist:
                    min_dist = dist
                    T = new_T
    pruning = 5 * T.number_of_nodes()  # how much do u wanna prune
    for i in range(pruning):
        random_node = random.choice(list(T.nodes()))
        t_new = T.copy()
        t_new.remove_node(random_node)
        if t_new.number_of_nodes() > 0 and is_valid_network(G, t_new) and average_pairwise_distance_fast(
                t_new) < average_pairwise_distance_fast(T):
            T = t_new
    return T

#returns shortest paths tree starting at source
def shortest_paths(G,source, path_type = 0):
    if path_type == 0:
        paths = nx.single_source_dijkstra_path(G,source)
    elif path_type == 1:
        paths = nx.single_source_bellman_ford_path(G, source)
    elif path_type == 2:
        paths = nx.shortest_path(G, source,method='bellman-ford')
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

# returns min shortest paths tree and corresponding distance
def min_shortest_paths(G):
    T = leaf_pruning(shortest_paths(G, 0))
    min_dist = average_pairwise_distance_fast(T)
    for source in G.nodes():
        new_T = leaf_pruning(shortest_paths(G, source))
        dist = average_pairwise_distance_fast(new_T)
        if dist < min_dist:
            min_dist = dist
            T = new_T
    return T, min_dist

def min_shortest_paths2(G):
    T = leaf_pruning(shortest_paths(G, 0,2))
    min_dist = average_pairwise_distance_fast(T)
    for source in G.nodes():
        new_T = leaf_pruning(shortest_paths(G, source,2))
        dist = average_pairwise_distance_fast(new_T)
        if dist < min_dist:
            min_dist = dist
            T = new_T
    return T, min_dist

def min_shortest_paths3(G):
    T = leaf_pruning(shortest_paths(G, 0))
    min_dist = average_pairwise_distance_fast(T)
    for source in G.nodes():
        new_T = dont_always_prune(shortest_paths(G, source))
        dist = average_pairwise_distance_fast(new_T)
        if dist < min_dist:
            min_dist = dist
            T = new_T
    return T, min_dist

# returns min spanning tree with leaves pruned. and corresponding distance
def min_spanning_tree(G):
    T_spanning = leaf_pruning(spanning_tree(G))
    dist_spanning = average_pairwise_distance_fast(T_spanning)
    return T_spanning, dist_spanning

#returns tree of min vertex cover with leaves pruned. and corresponding distance
def min_vertex_cover(G):
    nodes = nx.algorithms.approximation.min_weighted_vertex_cover(G)
    newG = G.subgraph(nodes)
    T = shortest_paths(newG, list(G.nodes())[0])
    min_dist = average_pairwise_distance_fast(T)
    for source in newG.nodes():
        new_T = shortest_paths(newG, source)
        dist = average_pairwise_distance_fast(new_T)
        if dist < min_dist:
            min_dist = dist
            T = new_T
    random_times = 10 # how much random????!!!
    min_dist = math.inf
    best_T = None
    for i in range(random_times):
        new_T = more_random_pruning(T)
        new_dist = average_pairwise_distance_fast(new_T)
        if new_dist < min_dist:
            min_dist = new_dist
            best_T = new_T
    return best_T, average_pairwise_distance_fast(best_T)

#continually adds most central nodes until tree is created
def min_centrality(G):
    centrality = {}
    for node in G.nodes():
        centrality[node] = nx.algorithms.centrality.closeness_centrality(G,node,distance='weight')
    nodes = []
    bcentrality = nx.algorithms.centrality.betweenness_centrality(G,weight='weight')
    for node in sorted(centrality,key=centrality.get,reverse=True):
        nodes += [node]
        newG = G.subgraph(nodes)
        T = shortest_paths(newG, list(newG.nodes())[0])
        if len(T.nodes()) > 0 and is_valid_network(G,T):
            min_dist = average_pairwise_distance_fast(T)
            for source in newG.nodes():
                new_T = shortest_paths(newG, source)
                dist = average_pairwise_distance_fast(new_T)
                if dist < min_dist:
                    min_dist = dist
                    T = new_T
            random_times = 10  # how much random????!!!
            min_dist = math.inf
            best_T = None
            for i in range(random_times):
                new_T = more_random_pruning(T)
                new_dist = average_pairwise_distance_fast(new_T)
                if new_dist < min_dist:
                    min_dist = new_dist
                    best_T = new_T
            return best_T, average_pairwise_distance_fast(best_T)

def steiner_tree(G):
    nodes = nx.dominating_set(G)
    T = nx.algorithms.approximation.steiner_tree(G,nodes,weight='weight')
    return T,average_pairwise_distance_fast(T)

def random_shortest_paths(G):
    V = G.nodes()
    E = G.edges(data=True)
    best_dist = math.inf
    best_tree = None
    for source in range(len(V)):
        for i in range(1000):
            paths = nx.single_source_dijkstra_path(G, source)
            # paths = nx.single_source_bellman_ford_path(G, source)
            # paths = nx.shortest_path(G, source,method='bellman-ford')
            edges = []

            for path in paths.values():
                # print(path)
                for i in range(len(path) - 1):
                    u = path[i]
                    v = path[i + 1]
                    if random.random() < 0.5:
                        edges += [(u, v, G.get_edge_data(u, v))]
                    else:
                        random_v = random.choice(list(G[u].keys()))
                        edges += [(u, random_v, G.get_edge_data(u, random_v))]
            # print(edges)
            T = leaf_pruning(nx.Graph(edges))
            if is_valid_network(G,T):
                dist = average_pairwise_distance_fast(T)
                if dist < best_dist:
                    best_dist = dist
                    best_tree = T
                print(dist)
    print("BEST:",best_dist)

def dominating_set(G):
    dset = list(nx.dominating_set(G))
    for source in dset:
        edges = []
        for node in dset:
            if node != source:
                shortest_path = nx.shortest_path(G,source,node)
                #add edges in path
                for i in range(len(shortest_path) - 1):
                    u = shortest_path[i]
                    v = shortest_path[i + 1]
                    edges += [(u, v, G.get_edge_data(u, v))]
        T = nx.Graph(edges)
        draw(G,T)
        print(average_pairwise_distance_fast(T))

def campos(G):
    E = G.edges()
    V = G.nodes()

    #initialize d, s, m, and sumWeights to 0
    d = [0 for _ in V]
    s = [0 for _ in V]
    m = [0 for _ in V]
    sumWeights = 0

    #computes degree and sum of weights for each vertex
    for i, j in E:
        weight = G[i][j]['weight']
        d[i], d[j] = d[i] + 1, d[j] + 1
        s[i], s[j] = s[i] + weight, s[j] + weight
        m[i], m[j] = max(m[i], weight), max(m[j], weight)
        sumWeights += weight

    #compute mean and std dev for the set of edge weights in G
    mean = sumWeights/len(E)
    the_sum = 0
    for i,j in E:
        weight = G[i][j]['weight']
        the_sum += (weight - mean)**2
    stdDev = (the_sum/(len(E)-1))**0.5
    ratio = stdDev/mean
    threshold = 0.4 + .005*(len(V) - 10)
    C4, C5 = 0, 0
    if ratio < threshold:
        C4, C5 = 1, 1
    else:
        C4, C5 = .9, .1

    #selects vertex with higher spanning potential as the initial vertex
    sp = [0 for _ in V] #have to initialize it or something
    sp_max = 0
    f = 0
    for v in V:
        sp[v] = 0.2*d[v] + 0.6*(d[v]/s[v]) + 0.2*(1/m[v])
        if sp[v] > sp_max:
            sp_max = sp[v]
            f = v

    #THE SHIT
    w = [math.inf for _ in V] # weight of vertex
    cf = [math.inf for _ in V] #estimate cost of path between v and f
    w[f], cf[f] = 0, 0
    pd = [None for _ in V]
    ps = [None for _ in V]
    pd[f], ps[f] = 0, 1
    p = [None for _ in V]
    p[f] = f
    L = {f}
    T = set()
    T_edges = []

    while L:
        wd_min = math.inf
        jsp_max = 0
        S = set()
        u = None
        for v in L:
            wd_v = C4*w[v] + C5*cf[v]
            if wd_v < wd_min:
                S = {v}
                wd_min = wd_v
            elif wd_v == wd_min:
                S.add(v)
        for v in S:
            jsp_v = (d[v]+pd[v]) + (d[v]+pd[v])/(s[v]+ps[v])
            if jsp_v >= jsp_max:
                jsp_max = jsp_v
                u = v
        for a in G[u]:
            if a in T:
                continue
            weight = G[u][a]['weight']
            wd_t = C4*weight + C5*(cf[u]+weight)
            jsp_t = (d[a]+d[u]) + (d[a]+d[u])/(s[a]+s[u])
            wd_a = C4*w[a] + C5*cf[a]
            jsp_a = None
            if pd[a] is None or ps[a] is None:
                jsp_a = 0
            else:
                jsp_a = (d[a] + pd[a]) + (d[a] + pd[a]) / (s[a] + ps[a])

            if wd_t < wd_a:
                w[a] = weight
                cf[a] = cf[u] + weight
                pd[a] = d[u]
                ps[a] = s[u]
                p[a] = u
            elif (wd_t == wd_a) and (jsp_t >= jsp_a):
                pd[a] = d[u]
                ps[a] = s[u]
                p[a] = u

            if a not in L:
                L.add(a)
        T.add(u)
        T.add(p[u])
        L.remove(u)
        if p[u] in L:
            L.remove(p[u])

        if u != p[u]:
            T_edges += [(u,p[u],G.get_edge_data(u,p[u]))]
    return nx.Graph(T_edges)

def jessica(G):
    seen = set()
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    top_fourth = [key for key,val in degrees[:len(G.nodes()) // 4]]
    print(top_fourth)
    for v in top_fourth:
        seen.add(v)
        for adj in G[v]:
            seen.add(adj)
    print(seen)
    for v, degree in degrees[len(G.nodes()) // 4 + 1:]:
        if len(seen) == len(G.nodes()):
            break
        top_fourth += [v]
        seen.add(v)
        for adj in G[v]:
            seen.add(adj)
    return top_fourth

def bruteforce(G):
    number_of_nodes = 15
    iterations = 10000
    global_min = math.inf
    best_tree = None
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    top_fourth = [key for key, val in degrees[:len(G.nodes()) - 6]]
    for i in range(8,number_of_nodes):
        for i in range(iterations):
            nodes = []
            for i in range(number_of_nodes):
                nodes += [random.choice(top_fourth)]
            subgraph = G.subgraph(nodes)
            if not is_valid_network(G, subgraph):
                continue
            min_dist = math.inf
            min_tree = None
            for node in subgraph.nodes():
                shortest_paths_tree = shortest_paths(subgraph,node)
                dist = average_pairwise_distance_fast(shortest_paths_tree)
                if dist < min_dist:
                    min_dist = dist
                    min_tree = shortest_paths_tree
            print(min_dist)
            if min_dist < global_min:
                global_min = min_dist
                best_tree = min_tree
        print("BEST:",global_min)
    print(best_tree.nodes())
    print(best_tree.edges())

def draw(G,T):
    pos = nx.spring_layout(G)
    nx.draw_networkx(G,pos=pos, node_color='blue')
    nx.draw_networkx(T, pos=pos, node_color='red', edge_color='red')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    #assert len(sys.argv) == 2
    start = 195
    end = 195
    size = 'small'
    draw_graph = True
    for i in range(start,1+end):
        path = 'inputs/' + size + '-'+str(i)+'.in'
        G = read_input_file(path)

        if G.number_of_nodes() <= 2:
            T = G.subgraph(0)
            write_output_file(T, 'outputs/' + size + '-' + str(i) + '.out')
            print(str(i), "is done.")
            continue

        functions = [min_shortest_paths, min_shortest_paths2, min_shortest_paths3, min_vertex_cover, min_centrality]
        min_dist = math.inf
        best_tree = None
        for func in functions:
            T,dist = func(G)
            #print(func,dist)
            #print("Tnodes",sorted(list(T.nodes())))
            if dist < min_dist and is_valid_network(G,T):
                min_dist = dist
                best_tree = T

        if draw_graph:
            draw(G,best_tree)
        assert is_valid_network(G, best_tree)
        print("Average pairwise distance: {}".format(average_pairwise_distance_fast(best_tree)))
        print(str(i),"is done.")
        write_output_file(best_tree, 'outputs/' + size + '-'+str(i)+'.out')
