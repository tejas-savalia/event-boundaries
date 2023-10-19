import os, itertools, multiprocessing
# os.environ['OMP_NUM_THREADS']="1"
import numpy as np
import pandas as pd
import scipy.stats as stat
import seaborn as sns
import matplotlib.pyplot as plt
# from hmmviz import TransGraphy
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import scipy


graph = np.zeros((15, 15))
def create_modular(graph):
    graph[0, (1, 2, 3, 14)] = 1
    graph[1, (0, 2, 3, 4)] = 1
    graph[2, (0, 1, 3, 4)] = 1
    graph[3, (0, 1, 2, 4)] = 1
    graph[4, (1, 2, 3, 5)] = 1
    graph[5, (4, 6, 7, 8)] = 1
    graph[6, (5, 7, 8, 9)] = 1
    graph[7, (5, 6, 8, 9)] = 1
    graph[8, (5, 6, 7, 9)] = 1
    graph[9, (6, 7, 8, 10)] = 1
    graph[10, (9, 11, 12, 13)] = 1
    graph[11, (10, 12, 13, 14)] = 1
    graph[12, (10, 11, 13, 14)] = 1
    graph[13, (10, 11, 12, 14)] = 1
    graph[14, (11, 12, 13, 0)] = 1
    return graph
# graph[0, 2] = 1
# graph[2, 0] = 1

# graph[1, 2] = 1

graph_lattice = np.zeros((15, 15))
def create_lattice(graph_lattice):
    graph_lattice[0, (1, 2, 3, 12)] = 1
    graph_lattice[1, (0, 2, 4, 13)] = 1
    graph_lattice[2, (0, 1, 5, 14)] = 1
    graph_lattice[3, (0, 4, 5, 6)] = 1
    graph_lattice[4, (1, 3, 5, 7)] = 1
    graph_lattice[5, (2, 3, 4, 8)] = 1
    graph_lattice[6, (3, 7, 8, 9)] = 1
    graph_lattice[7, (4, 6, 8, 10)] = 1
    graph_lattice[8, (5, 6, 7, 11)] = 1
    graph_lattice[9, (6, 10, 11, 12)] = 1
    graph_lattice[10, (7, 9, 11, 13)] = 1
    graph_lattice[11, (8, 9, 10, 14)] = 1
    graph_lattice[12, (9, 13, 14, 0)] = 1
    graph_lattice[13, (10, 12, 14, 1)] = 1
    graph_lattice[14, (10, 11, 13, 2)] = 1
    return graph_lattice


def random_walk(graph):
    #Random Walk
    start_state = np.random.choice(range(8))
    path_length = 1000
    current_state = start_state
    path = np.zeros(path_length)
    for i in range(path_length):
        path[i] = current_state
        neighbour_states = np.where(graph[current_state])[0]
        next_state = np.random.choice(neighbour_states)
        current_state = next_state
    return path
        

def draw_SR_categories(path, cutoff_point, alpha = 0.1, gamma = 0.1, num_nodes = 15, plot = True):
    SR = np.random.uniform(0, 1, size=(num_nodes, num_nodes))
    start_state = np.random.choice(np.arange(8))
    current_state = start_state    
    cmap = plt.cm.rainbow


    for i, observed_state in enumerate(path[:cutoff_point]):

        expected_probs = SR[current_state]
        one_hot_obs = np.zeros(num_nodes)
        one_hot_obs[int(observed_state)] = 1

        SR_delta = one_hot_obs + gamma*SR[int(observed_state), :] - expected_probs

        SR[current_state, :] = SR[current_state, :] + alpha*SR_delta
        SR[current_state, :] = SR[current_state, :]/sum(SR[current_state])
        current_state = int(observed_state)    



    G = nx.Graph() 


    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(SR)
#     gm = GaussianMixture(n_components=3, random_state=0, reg_covar=0.05).fit(SR)
    node_color = []
    node_transparency = []
    clusters_assigned = kmeans.predict(SR)
    centroids = kmeans.cluster_centers_
    
#     print(distances)

#     print(prob, cluster)
#     print(prob)
    for node in range(15):
        node_distance = np.linalg.norm(SR[node] - centroids, axis=1)

#         node_confidences = np.exp(-node_distance / 2) / np.sum(np.exp(-node_distance / 2), axis=0)
        node_confidences = scipy.special.softmax(1/node_distance)
        node_transparency.append(node_confidences)
        
        

        if kmeans.labels_[node] == 0:
#         if cluster[node] == 0:
            node_color.append('red')
        elif kmeans.labels_[node] == 1:
#         elif cluster[node] == 1:
            node_color.append('green')
        else:
            node_color.append('blue')
        
        G.add_node(str(node))

    # print(node_transparency)

    if plot:
        for i in range(SR.shape[0]):
            for j in range(SR.shape[1]):
                G.add_edge(str(i), str(j), weight = SR[i][j])    
                
        edges = G.edges
        weights = [SR[int(u)][int(v)] for u,v in edges]

        node_pos = nx.spring_layout(G)
        node_labels = np.arange(15).astype(str)

        nx.draw_networkx_nodes(G, pos=node_pos, node_color=node_transparency,  nodelist=G.nodes)
        nx.draw_networkx_edges(G, pos=node_pos, width=weights)
        nx.draw_networkx_labels(G, pos = node_pos)
        
        nx.draw(G, node_color=node_color, width = weights, with_labels = True, alpha=0.5)#, alpha = node_transparency)
    
    # plt.show()
    return SR

def compute_entropies(params):
    entropy = np.zeros(100)
    graph = np.zeros((15, 15))

    alpha = params[0]
    gamma = params[1]
    if params[2] == 'modular':
        graph = create_modular(graph)
    else:
        graph = create_lattice(graph)

    for e in range(100):
        path = random_walk(graph)
        SR = draw_SR_categories(path, 1000, alpha=alpha, gamma=gamma, plot=False)
        entropy[e] = -np.sum(SR*np.log(SR))
        if e%99 == 0:
            print(entropy[e])
    return entropy


params = itertools.product([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99], [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99], ['modular', 'lattice'])

p = multiprocessing.Pool()
entropy = p.map(compute_entropies, params)

params = itertools.product([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99], [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99], ['modular', 'lattice'])
params = np.array([a for a in params])

df_entropy = pd.DataFrame({
    'alpha': np.repeat(params[:, 0], 100),
    'gamma': np.repeat(params[:, 1], 100),
    'graph type': np.repeat(params[:, 2], 100),
    'entropy': np.ravel(entropy)
})

print(df_entropy)
df_entropy.to_csv('results/df_entropy.csv', index = False)