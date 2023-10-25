import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import itertools, multiprocessing
import scipy.stats as stat
from sklearn.cluster import KMeans


def create_adjacency_matrix_for_modular_graph(num_nodes, num_modules, module_sizes, inter_module_edges, boundary_nodes):
  """
  Creates an adjacency matrix for a graph with modular structure.

  Args:
    num_nodes: The total number of nodes in the graph.
    num_modules: The number of modules in the graph.
    module_sizes: A list of the sizes of each module.
    inter_module_edges: A list of edges between modules.

  Returns:
    An adjacency matrix for the graph.
  """

  # Create an empty adjacency matrix.
  adj_matrix = np.zeros((num_nodes, num_nodes))

  # Add edges within each module.
  for module_index in range(num_modules):
    module_start_index = sum(module_sizes[:module_index])
    module_end_index = module_start_index + module_sizes[module_index]

    for node_index in range(module_start_index, module_end_index):
      for other_node_index in range(module_start_index, module_end_index):
        if node_index != other_node_index:
          adj_matrix[node_index, other_node_index] = 1
        
  for node_i in boundary_nodes:
    for node_j in boundary_nodes:
      adj_matrix[node_i][node_j] = 0

  # Add edges between modules.
  for edge in inter_module_edges:
    node_index_1, node_index_2 = edge
    adj_matrix[node_index_1, node_index_2] = 1
    adj_matrix[node_index_2, node_index_1] = 1

  return adj_matrix

def random_walk(graph, path_length = 1000):
    #Random Walk
    start_state = np.random.choice(range(graph.shape[0]))
    current_state = start_state
    path = np.zeros(path_length)
    for i in range(path_length):
        path[i] = current_state
        neighbour_states = np.where(graph[current_state])[0]
        next_state = np.random.choice(neighbour_states)
        current_state = next_state
    return path
        

def random_hop(graph, hop_step = 1, path_length = 1000):
    #Random Walk
    start_state = np.random.choice(range(graph.shape[0]))
    current_state = start_state
    path = np.zeros(path_length)

    for i in range(path_length):

        if i%hop_step == 0:
            start_state = np.random.choice(range(graph.shape[0]))
            current_state = start_state

        neighbour_states = np.where(graph[current_state])[0]
        next_state = np.random.choice(neighbour_states)
        path[i] = current_state
        current_state = next_state

    return path


def run_SR(path, graph, alpha = 0.1, gamma = 0.1, plot = True):
    SR = np.random.uniform(0, 1, size=graph.shape)
    num_nodes = graph.shape[0]
    start_state = np.random.choice(np.arange(num_nodes))
    current_state = start_state    


    for observed_state in path:

        expected_probs = SR[current_state]
        one_hot_obs = np.zeros(num_nodes)
        one_hot_obs[int(observed_state)] = 1

        SR_delta = one_hot_obs + gamma*SR[int(observed_state), :] - expected_probs

        SR[current_state, :] = SR[current_state, :] + alpha*SR_delta
        SR[current_state, :] = SR[current_state, :]/sum(SR[current_state])
        current_state = int(observed_state)

    if plot:
        sns.heatmap(SR)
    return SR

def compute_node_entropies(params):
    alpha = params[0]
    gamma = params[1]
    num_nodes = 15
    num_modules = 3
    boundary_nodes = [x for x in range(num_nodes) if ((x%5 == 0) or (x%5 == 4))]
    crossmodule_connections = [(0, 14), (4, 5), (9, 10)]
    node_entropy = np.zeros(num_nodes)

    graph = create_adjacency_matrix_for_modular_graph(num_nodes, num_modules, np.repeat(num_nodes//num_modules, num_modules), 
                                                      crossmodule_connections, boundary_nodes)
    if len(params)>2:
        if params[2] == 'hop':
            SR = run_SR(path=random_hop(graph, hop_step=params[3]), graph=graph, alpha=alpha, gamma=gamma, plot=False)
        else:
            SR = run_SR(path=random_walk(graph), graph=graph, alpha=alpha, gamma=gamma, plot=False)

            

    # graph_entropy = -np.sum(SR*np.log(SR))
    for node in range(graph.shape[0]):
        node_entropy[node] = -np.sum(SR[node]*np.log(SR[node]))

    return node_entropy


if __name__ == '__main__':
    p = multiprocessing.Pool()
    params = list(itertools.product([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9], [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9], ['walk', 'hop'], [1, 2, 3, 4, 5, 6]))
    print(np.array(params)[:, 0])
    node_entropies = np.array([p.map(compute_node_entropies, params) for _ in range(100)])

    df_node_entropies = pd.DataFrame({
    'alpha': np.tile(np.repeat(np.array(params)[:, 0], 15), 100),
    'gamma': np.tile(np.repeat(np.array(params)[:, 1], 15), 100),
    'walk type': np.tile(np.repeat(np.array(params)[:, 2], 15), 100),
    'hope length': np.tile(np.repeat(np.array(params)[:, 3], 15), 100),
    'iteration': np.repeat(np.arange(100), len(params)*15),
    'node entropies': np.ravel(node_entropies)
    })

    print(df_node_entropies)
    df_node_entropies.to_csv('modular_node_entropies.csv')