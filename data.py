import numpy as np
import pickle

test_data = '../test_data/test_data.pkl'

def pad_graph(features, coordinates):
    """ Pads a set of graphs with zeros to fit the size of the largest graph. 
    
    Parameters:
    -----------
    features : list
        A list of features ndarrays with shape [N, D].
    coordinates : list
        A list of coordinate ndarrays with shape [N, 3].
    
    Returns:
    --------
    features_padded : ndarray, shape [n, N_max, D]
        Feature matrices for n graphs.
    coordinates_padded : ndarray, shape [n, N_max, 3]
        Coordinate matrices for n graphs.
    masks : ndarray, shape [n, N_max]
        Masks for n graphs, indicating if a node was in the original graph.
    """
    N_max = max(map(lambda ndarray: ndarray.shape[0], features))
    N_graphs = len(features)
    features_padded = np.zeros((N_graphs, N_max, features[0].shape[1]))
    coordinates_padded = np.zeros((N_graphs, N_max, 3))
    masks = np.zeros((N_graphs, N_max))
    print(f'Padding to {N_max} vertices.')
    for graph_idx, (f, c) in enumerate(zip(features, coordinates)):
        features_padded[graph_idx, :f.shape[0], :] = f
        coordinates_padded[graph_idx, :f.shape[0], :] = c
        masks[graph_idx, :f.shape[0]] = 1.0
    return features_padded, coordinates_padded, masks

def load_padded_test_data(n_graphs_per_class, pickle_path = test_data, shuffle=True):
    """ Loads and pads a portion of the test data into memory.
    
    Parameters:
    -----------
    n_graphs_per_class : int
        Number of graphs to load per class.
    pickle_path : str
        Path to the pickle file that contains test data.
    shuffle : bool
        If the data will be shuffled.
    
    Returns:
    --------
    features_padded : ndarray, shape [n, N_max, D]
        Feature matrices for n graphs.
    coordinates_padded : ndarray, shape [n, N_max, 3]
        Coordinate matrices for n graphs.
    masks : ndarray, shape [n, N_max]
        Masks for n graphs, indicating if a node was in the original graph.
    targets : ndarray, shape [n]
        Class labels.
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    features, coordinates = [], []
    targets = []
    for class_label, interaction_type in enumerate((b'numu', b'nue', b'nutau')):
        has_track = interaction_type == b'numu'
        n_graphs = n_graphs_per_class if has_track else n_graphs_per_class // 2 # Remove class imbalance
        features += data[interaction_type][b'features'][:n_graphs]
        coordinates += data[interaction_type][b'coordinates'][:n_graphs]
        #targets += [class_label] * n_graphs
        targets += [int(has_track)] * n_graphs
        # print(interaction_type, coordinates[0].shape, len(features), len(coordinates), features[0].shape)
    features_padded, coordinates_padded, masks = pad_graph(features, coordinates)
    targets = np.array(targets)
    if shuffle:
        idxs = np.arange(features_padded.shape[0])
        np.random.shuffle(idxs)
        features_padded = features_padded[idxs].squeeze()
        coordinates_padded = coordinates_padded[idxs].squeeze()
        masks = masks[idxs].squeeze()
        targets = targets[idxs].squeeze()
    return features_padded, coordinates_padded, masks, targets
