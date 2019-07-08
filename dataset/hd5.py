import numpy as np
import pickle
import h5py
from collections import defaultdict
import tempfile
import os
import hashlib
from sklearn.metrics import pairwise_distances
from sklearn.utils.class_weight import compute_sample_weight

import torch.utils.data

class ShuffledTorchHD5Dataset(torch.utils.data.Dataset):
    """ Class to represent a pre-shuffled PyTorch dataset originating from an HD5File. """

    def __init__(self, filepath, features=['CumulativeCharge', 'Time', 'FirstCharge'], coordinates=['VertexX', 'VertexY', 'VertexZ'], 
        balance_dataset=False, min_track_length=None, max_cascade_energy=None, flavors=None, currents=None,
        memmap_directory='./memmaps', close_file=True, 
        class_weights=None):
        """ Initializes the Dataset. 
        
        Parameters:
        -----------
        filepath : str
            A path to the hd5 file.
        features : list
            A list of dataset columns in the HD5 File that represent the vertex features.
        coordinates : list
            A list of coordinate columns in the HD5 File that represent the coordinates of each vertex.
        balance_dataset : bool
            If the dataset should be balanced such that each class contains the same number of samples.
        min_track_length : float or None
            Minimal track length all track-events must have.
        max_cascade_energy : float or None
            The maximal cascade energy all track events are allowed to have.
        flavors : list or None
            Only certain neutrino flavor events will be considered if given.
        currents : list or None
            Only certain current events will be considered if given. 
        memmap_directory : str
            Directory for memmaps.
        close_file : bool
            If True, the hd5 file will be closed afterwards.
        class_weights : dict or None
            Weights for each class.
        """
        self.file = h5py.File(filepath)
        self.feature_names = features
        self.coordinate_names = coordinates

        # Create class targets
        interaction_type = np.array(self.file['InteractionType'], dtype=np.int8)
        pdg_encoding = np.array(self.file['PDGEncoding'], dtype=np.int8)
        has_track = np.logical_and(np.abs(pdg_encoding) == 14, interaction_type == 1)
        targets = has_track.astype(np.int)

        number_vertices = np.array(self.file['NumberVertices'])
        event_offsets = number_vertices.cumsum() - number_vertices

        # Apply filters to the dataset
        filter = event_filter(self.file, min_track_length=min_track_length, max_cascade_energy=max_cascade_energy, flavors=flavors, currents=currents)
        if balance_dataset:
            # Initialize numpys random seed with a hash of the filepath such that always the same indices get chosen 'randomly'
            np.random.seed(int(hashlib.sha1(filepath.encode()).hexdigest(), 16) & 0xFFFFFFFF)
            classes, class_counts = np.unique(targets[filter], return_counts=True)
            print(f'Classes {classes}; Class counts {class_counts}')
            min_class_size = np.min(class_counts)
            for class_ in classes:
                # Remove the last samples of the larger class
                class_idx = np.where(np.logical_and((targets == class_), filter))[0]
                np.random.shuffle(class_idx)
                filter[class_idx[min_class_size : ]] = False
            self._idxs = np.where(filter)[0]
            # Assert that the class counts are the same now
            _, class_counts = np.unique(targets[self._idxs], return_counts=True)
            assert np.allclose(class_counts, min_class_size)
            print(f'Reduced dataset to {min_class_size} samples per class ({self._idxs.shape[0]} / {targets.shape[0]})')
        else:
            self._idxs = np.where(filter)[0]
        print(number_vertices.shape, self._idxs.shape)
        self.number_vertices = number_vertices[self._idxs]
        self.event_offsets = self.number_vertices.cumsum() - self.number_vertices
        self.targets = targets[self._idxs]

        # Create memmaps for features and coordinates for faster access during training
        os.makedirs(os.path.dirname(memmap_directory), exist_ok=True)

        # Load precomputed memmaps based on the hash of the columns, filename and index set
        self._idxs_hash = hashlib.sha1(self._idxs.data).hexdigest()
        features_hash = hashlib.sha1(str([
            [os.path.relpath(filepath)] + features
        ]).encode()).hexdigest()
        coordinates_hash = hashlib.sha1(str([
            [os.path.relpath(filepath)] + coordinates
        ]).encode()).hexdigest()

        feature_memmap_path = os.path.join(memmap_directory, f'hd5_features_{features_hash}_{self._idxs_hash}')
        coordinate_memmap_path = os.path.join(memmap_directory, f'hd5_coordinates_{coordinates_hash}_{self._idxs_hash}')

        if not os.path.exists(feature_memmap_path) or not os.path.exists(coordinate_memmap_path):
            # Create an index set that operates on vertex features, which is used to build memmaps efficiently
            _vertex_idxs = np.concatenate([np.arange(start, end) for start, end in zip(self.event_offsets, self.event_offsets + self.number_vertices)]).tolist()
        
        if not os.path.exists(feature_memmap_path):
            # Create a new memmap for all features
            self.features = np.memmap(feature_memmap_path, shape=(self.number_vertices.sum(), len(self.feature_names)), dtype=np.float64, mode='w+')
            for feature_idx, feature in enumerate(self.feature_names):
                print(f'\rCreating column for feature {feature}', end='\r')
                self.features[:, feature_idx] = np.array(self.file.get(feature))[_vertex_idxs]
            print(f'\nCreated feature memmap {feature_memmap_path}')
        else:
            self.features = np.memmap(feature_memmap_path, shape=(self.number_vertices.sum(), len(self.feature_names)), dtype=np.float64)

        if not os.path.exists(coordinate_memmap_path):
            # Create a new memmap for all features
            self.coordinates = np.memmap(coordinate_memmap_path, shape=(self.number_vertices.sum(), len(self.coordinate_names)), dtype=np.float64, mode='w+')
            for coordinate_idx, coordinate in enumerate(self.coordinate_names):
                print(f'\rCreating column for coordinate {coordinate}', end='\r')
                self.coordinates[:, coordinate_idx] = np.array(self.file.get(coordinate))[_vertex_idxs]
            print(f'\nCreated coordinate memmap {coordinate_memmap_path}')
        else:
            self.coordinates = np.memmap(coordinate_memmap_path, shape=(self.number_vertices.sum(), len(self.coordinate_names)), dtype=np.float64)

        # Compute weights, if a dictionary is given, string keys must be converted to int keys
        if isinstance(class_weights, dict):
            class_weights = {int(class_) : weight for class_, weight in class_weights.items()}
        self.weights = compute_sample_weight(class_weights, self.targets)

        # Sanity checks
        endpoints = self.number_vertices + self.event_offsets
        assert(np.max(endpoints)) <= self.features.shape[0]
        # print(np.unique(np.vstack((np.abs(pdg_encoding[self._idxs]), interaction_type[self._idxs])), return_counts=True, axis=1))

        if close_file:
            self.file.close()
            self.file = filepath

    def __len__(self):
        return self.number_vertices.shape[0]
    
    def __getitem__(self, idx):
        N = self.number_vertices[idx]
        offset = self.event_offsets[idx]
        X = self.features[offset : offset + N]
        C = self.coordinates[offset : offset + N]
        return X, C, self.targets[idx], self.weights[idx]

    @staticmethod
    def collate(samples):
        """ Collator for the dataset wrapper.
        
        Parameters:
        -----------
        samples : list
            A list of tuples representing the different inputs and targets.
        
        Returns:
        --------
        inputs : tuple
            - X : torch.FloatTensor, shape [batch_size, N, D]
                The vertex features.
            - C : torch.FloatTensor, shape [batch_size, N, num_coordinates]
                Pariwise distances.
            - masks : torch.FloatTensor, shape [batch_size, N, N]
                Adjacency matrix masks.
        outputs : tuple
            - torch.FloatTensor, shape [batch_size, N, 1]
                Class labels.
        weights : torch.FloatTensor, shape [batch_size] or None
            Weights for each sample.
        """
        X, C, y, w = zip(*samples)

        # Pad the batch
        batch_size = len(X)
        max_number_vertices = max(map(lambda features: features.shape[0], X))
        features = np.zeros((batch_size, max_number_vertices, X[0].shape[1]))
        coordinates = np.zeros((batch_size, max_number_vertices, C[0].shape[1]))
        masks = np.zeros((batch_size, max_number_vertices, max_number_vertices))
        for idx, (X_i, C_i) in enumerate(zip(X, C)):
            features[idx, : X_i.shape[0]] = X_i
            coordinates[idx, : C_i.shape[0], : C_i.shape[1]] = C_i
            masks[idx, : X_i.shape[0], : X_i.shape[0]] = 1
        
        # Make torch tensors
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        features = torch.FloatTensor(features).to(device)
        coordinates = torch.FloatTensor(coordinates).to(device)
        masks = torch.FloatTensor(masks).to(device)
        targets = torch.FloatTensor(y).to(device).unsqueeze(1)
        weights = torch.FloatTensor(w).to(device).unsqueeze(1)
        return (features, coordinates, masks), targets, weights

class ShuffledTorchHD5DatasetWithGraphFeatures(ShuffledTorchHD5Dataset):
    """ Pre-shuffled PyTorch dataset that also outputs graph features. """
    def __init__(self, filepath, features=['CumulativeCharge', 'Time', 'FirstCharge'], coordinates=['VertexX', 'VertexY', 'VertexZ'],
        graph_features = ['RecoX', 'RecoY', 'RecoZ', 'RecoAzimuth', 'RecoZenith'], 
        balance_dataset=False, min_track_length=None, max_cascade_energy=None, flavors=None, currents=None,
        memmap_directory='./memmaps', close_file=True, class_weights=None):
        """ Initializes the Dataset. 
        
        Parameters:
        -----------
        filepath : str
            A path to the hd5 file.
        features : list
            A list of dataset columns in the HD5 File that represent the vertex features.
        coordinates : list
            A list of coordinate columns in the HD5 File that represent the coordinates of each vertex.
        graph_features : list
            A list of graph feature columns in the HD5 File that represent event features.
        balance_dataset : bool
            If the dataset should be balanced such that each class contains the same number of samples.
        min_track_length : float or None
            Minimal track length all track-events must have.
        max_cascade_energy : float or None
            The maximal cascade energy all track events are allowed to have.
        flavors : list or None
            Only certain neutrino flavor events will be considered if given.
        currents : list or None
            Only certain current events will be considered if given. 
        memmap_directory : str
            Directory for memmaps.
        close_file : bool
            If True, the hd5 file will be closed after readout.
        class_weights : dict or None
            Weights for each class.
        """
        super().__init__(filepath, features=features, coordinates=coordinates, balance_dataset=balance_dataset, 
            min_track_length=min_track_length, max_cascade_energy=max_cascade_energy, flavors=flavors, currents=currents,
            memmap_directory=memmap_directory, 
            close_file=False, class_weights=class_weights)
        self.graph_feature_names = graph_features

        graph_features_hash = hashlib.sha1(str([
            [os.path.relpath(filepath)] + graph_features
        ]).encode()).hexdigest()

        graph_features_memmap_path = os.path.join(memmap_directory, f'hd5_graph_features_{graph_features_hash}_{self._idxs_hash}')

        if not os.path.exists(graph_features_memmap_path):
            # Create a new memmap for all graph features
            self.graph_features = np.memmap(graph_features_memmap_path, shape=(self._idxs.shape[0], len(self.graph_feature_names)), dtype=np.float64, mode='w+')
            for feature_idx, graph_feature in enumerate(self.graph_feature_names):
                print(f'\rCreating column for graph_feature {graph_feature}', end='\r')
                self.graph_features[:, feature_idx] = np.array(self.file.get(graph_feature))[self._idxs]
            print(f'\nCreated feature memmap {graph_features_memmap_path}')
        else:
            self.graph_features = np.memmap(graph_features_memmap_path, shape=(self._idxs.shape[0], len(self.graph_feature_names)), dtype=np.float64)

        if close_file:
            self.file.close()
            self.file = filepath

    def __getitem__(self, idx):
        X, C, y, w = super().__getitem__(idx)
        return X, C, self.graph_features[idx], y, w

    @staticmethod
    def collate(samples):
        """ Collator for the dataset wrapper.
        
        Parameters:
        -----------
        samples : list
            A list of tuples representing the different inputs and targets.
        
        Returns:
        --------
        inputs : tuple
            - X : torch.FloatTensor, shape [batch_size, N, D]
                The vertex features.
            - C : torch.FloatTensor, shape [batch_size, N, num_coordinates]
                Pariwise distances.
            - masks : torch.FloatTensor, shape [batch_size, N, N]
                Adjacency matrix masks.
            - F : torch.FloatTensor, shape [batch_size, N]
                Graph features.
        outputs : tuple
            - targets : torch.FloatTensor, shape [batch_size, N, 1]
                Class labels.
        weights : torch.FloatTensor, shape [batch_size] or None
            Sample weights
        """
        X, C, F, y, w = zip(*samples)

        # Pad the batch
        batch_size = len(X)
        max_number_vertices = max(map(lambda features: features.shape[0], X))
        features = np.zeros((batch_size, max_number_vertices, X[0].shape[1]))
        coordinates = np.zeros((batch_size, max_number_vertices, C[0].shape[1]))
        masks = np.zeros((batch_size, max_number_vertices, max_number_vertices))
        graph_features = np.array(F)
        for idx, (X_i, C_i) in enumerate(zip(X, C)):
            features[idx, : X_i.shape[0]] = X_i
            coordinates[idx, : C_i.shape[0], : C_i.shape[1]] = C_i
            masks[idx, : X_i.shape[0], : X_i.shape[0]] = 1
        
        # Make torch tensors
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        features = torch.FloatTensor(features).to(device)
        coordinates = torch.FloatTensor(coordinates).to(device)
        masks = torch.FloatTensor(masks).to(device)
        targets = torch.FloatTensor(y).to(device).unsqueeze(1)
        graph_features = torch.FloatTensor(graph_features).to(device)
        weights = torch.FloatTensor(w).to(device).unsqueeze(1)
        return (features, coordinates, masks, graph_features), targets, weights

class ShuffledTorchHD5DatasetWithGraphFeaturesAndAuxiliaryTargets(ShuffledTorchHD5DatasetWithGraphFeatures):
    """ Pre-shuffled PyTorch dataset that will yield graph features as well as regression targets for an auxilliary task. """
    def __init__(self, filepath, features=['CumulativeCharge', 'Time', 'FirstCharge'], coordinates=['VertexX', 'VertexY', 'VertexZ'],
        graph_features = ['RecoX', 'RecoY', 'RecoZ', 'RecoAzimuth', 'RecoZenith'], auxiliary_targets=[],
        balance_dataset=False, min_track_length=None, max_cascade_energy=None, flavors=None, currents=None,
        memmap_directory='./memmaps', close_file=True, class_weights=None):
        """ Initializes the Dataset. 
        
        Parameters:
        -----------
        filepath : str
            A path to the hd5 file.
        features : list
            A list of dataset columns in the HD5 File that represent the vertex features.
        coordinates : list
            A list of coordinate columns in the HD5 File that represent the coordinates of each vertex.
        graph_features : list
            A list of graph feature columns in the HD5 File that represent event features.
        auxiliary_targets : list
            A list of graph feature columns that are used as targets for the auxiliary regression task.
        balance_dataset : bool
            If the dataset should be balanced such that each class contains the same number of samples.
        min_track_length : float or None
            Minimal track length all track-events must have.
        max_cascade_energy : float or None
            The maximal cascade energy all track events are allowed to have.
        flavors : list or None
            Only certain neutrino flavor events will be considered if given.
        currents : list or None
            Only certain current events will be considered if given. 
        memmap_directory : str
            Directory for memmaps.
        close_file : bool
            If True, the hd5 will be closed after readout.
        class_weights : dict or None
            Weights for each class.
        """
        super().__init__(filepath, features=features, coordinates=coordinates, balance_dataset=balance_dataset, 
            min_track_length=min_track_length, max_cascade_energy=max_cascade_energy, flavors=flavors, currents=currents,
            memmap_directory=memmap_directory, 
            graph_features=graph_features, close_file=False, class_weights=class_weights)

        self.auxiliary_target_names = auxiliary_targets

        auxiliary_targets_hash = hashlib.sha1(str([
            [os.path.relpath(filepath)] + auxiliary_targets
        ]).encode()).hexdigest()

        auxiliary_targets_memmap_path = os.path.join(memmap_directory, f'hd5_auxiliary_targets_{auxiliary_targets_hash}_{self._idxs_hash}')

        if not os.path.exists(auxiliary_targets_memmap_path):
            # Create a new memmap for all graph features
            self.auxiliary_targets = np.memmap(auxiliary_targets_memmap_path, shape=(self._idxs.shape[0], len(self.auxiliary_target_names)), dtype=np.float64, mode='w+')
            for target_idx, auxilary_target in enumerate(self.auxiliary_target_names):
                print(f'\rCreating column for auxiliary target {auxiliary_target}', end='\r')
                self.auxiliary_targets[:, target_idx] = np.array(self.file.get(auxilary_target))[self._idxs]
            print(f'\nCreated auxiliary target memmap {auxiliary_targets_memmap_path}')
        else:
            self.auxiliary_targets = np.memmap(auxiliary_targets_memmap_path, shape=(self._idxs.shape[0], len(self.auxiliary_target_names)), dtype=np.float64)

        if close_file:
            self.file.close()
            self.file = filepath

    def __getitem__(self, idx):
        X, C, F, y, w = super().__getitem__(idx)
        return X, C, F, y, self.auxiliary_targets[idx], w

    
    @staticmethod
    def collate(samples):
        """ Collator for the dataset wrapper.
        
        Parameters:
        -----------
        samples : list
            A list of tuples representing the different inputs and targets.
        
        Returns:
        --------
        inputs : tuple
            - X : torch.FloatTensor, shape [batch_size, N, D]
                The vertex features.
            - C : torch.FloatTensor, shape [batch_size, N, num_coordinates]
                Pariwise distances.
            - masks : torch.FloatTensor, shape [batch_size, N, N]
                Adjacency matrix masks.
            - F : torch.FloatTensor, shape [batch_size, N]
                Graph features.
        targets : tuple
            - targets : tuple
                - y : torch.FloatTensor, shape [batch_size, 1]
                    Class labels.
                - r : torch.FloatTensor, shape [batch_size, K]
                    Regression targets.
            - w : torch.FloatTensor, shape [batch_size]
                Sample weights.
        """
        X, C, F, y, r, w  = zip(*samples)

        # Pad the batch
        batch_size = len(X)
        max_number_vertices = max(map(lambda features: features.shape[0], X))
        features = np.zeros((batch_size, max_number_vertices, X[0].shape[1]))
        coordinates = np.zeros((batch_size, max_number_vertices, C[0].shape[1]))
        masks = np.zeros((batch_size, max_number_vertices, max_number_vertices))
        graph_features = np.array(F)
        for idx, (X_i, C_i) in enumerate(zip(X, C)):
            features[idx, : X_i.shape[0]] = X_i
            coordinates[idx, : C_i.shape[0], : C_i.shape[1]] = C_i
            masks[idx, : X_i.shape[0], : X_i.shape[0]] = 1
        
        # Make torch tensors
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        features = torch.FloatTensor(features).to(device)
        coordinates = torch.FloatTensor(coordinates).to(device)
        masks = torch.FloatTensor(masks).to(device)
        targets = torch.FloatTensor(y).to(device).unsqueeze(1)
        regression_targets = torch.FloatTensor(r).to(device)
        graph_features = torch.FloatTensor(graph_features).to(device)
        weights = torch.FloatTensor(w).to(device).unsqueeze(1)
        return (features, coordinates, masks, graph_features), (targets, regression_targets), weights

def event_filter(file, min_track_length=None, max_cascade_energy=None, min_total_energy=None, max_total_energy=None, 
    flavors=None, currents=None):
    """ Filters events by certain requiremenents.
    
    Parameters:
    -----------
    file : h5py.File
        The file from which to extract the attributes for each event.
    min_track_length : float or None
        All events with a track length lower than this will be excluded
        (events with no track will not be removed).
    max_cascade_energy : float or None
        All events with a track length that is not nan will be excluded
        if their cascade energy exceeds that threshold.
    min_total_energy : float or None
        All events with a total energy (cascade + muon) less than that will be excluded.
    max_total_energy : float or None
        All events with a total energy (cascade + muon) more than that will be excluded.
    flavors : list or None
        Only certain neutrino flavor events will be considered if given.
    currents : list or None
        Only certain current events will be considered if given. 

    Returns:
    --------
    filter : ndarray, shape [N], dtype=np.bool
        Only events that passed all filters are masked with True.
    """
    track_length = np.array(file['TrackLength'])
    cascade_energy = np.array(file['CascadeEnergy'])
    muon_energy = np.array(file['MuonEnergy'])
    muon_energy[np.isnan(muon_energy)] = 0
    total_energy = cascade_energy.copy()
    total_energy[np.isnan(total_energy)] = 0
    total_energy += muon_energy

    filter = np.ones(track_length.shape[0], dtype=np.bool)
    has_track_length = ~np.isnan(track_length)

    # Track length filter
    if min_track_length is not None:
        idx_removed = np.where(np.logical_and((track_length < min_track_length), has_track_length))[0]
        filter[idx_removed] = False
        print(f'After Track Length filter {filter.sum()} / {filter.shape[0]} events remain.')
    
    # Cascade energy filter
    if max_cascade_energy is not None:
        idx_removed = np.where(np.logical_and((cascade_energy > max_cascade_energy), has_track_length))
        filter[idx_removed] = False
        print(f'After Cascade Energy filter {filter.sum()} / {filter.shape[0]} events remain.')
    
    # Flavor filter
    if flavors is not None:
        pdg_encoding = np.array(file['PDGEncoding'])
        flavor_mask = np.zeros_like(filter, dtype=np.bool)
        for flavor in flavors:
            flavor_mask[np.abs(pdg_encoding) == flavor] = True
        filter = np.logical_and(filter, flavor_mask)
        print(f'After Flavor filter {filter.sum()} / {filter.shape[0]} events remain.')

    # Current filter
    if currents is not None:
        interaction_type = np.array(file['InteractionType'])
        current_mask = np.zeros_like(filter, dtype=np.bool)
        for current in currents:
            current_mask[np.abs(interaction_type) == current] = True
        filter = np.logical_and(filter, current_mask)
        print(f'After Current filter {filter.sum()} / {filter.shape[0]} events remain.')

    # Min total nergy filter
    if min_total_energy is not None:
        idx_removed = np.where(total_energy < min_total_energy)[0]
        filter[idx_removed] = False
        print(f'After Min Total Energy filter {filter.sum()} / {filter.shape[0]} events remain.')

    # Max total nergy filter
    if max_total_energy is not None:
        idx_removed = np.where(total_energy > max_total_energy)[0]
        filter[idx_removed] = False
        print(f'After Max Total Energy filter {filter.sum()} / {filter.shape[0]} events remain.')

    return filter