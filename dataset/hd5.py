import numpy as np
import pickle
import h5py
from collections import defaultdict
import tempfile
import os
import hashlib
from sklearn.metrics import pairwise_distances

import torch.utils.data


class ShuffledTorchHD5Dataset(torch.utils.data.Dataset):
    """ Class to represent a pre-shuffled PyTorch dataset originating from an HD5File. """

    def __init__(self, filepath, features=['CumulativeCharge', 'Time', 'FirstCharge'], coordinates=['VertexX', 'VertexY', 'VertexZ'], 
        balance_dataset=False, min_track_length=None, max_cascade_energy=None, memmap_directory='./memmaps'):
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
        memmap_directory : str
            Directory for memmaps.
        """
        self.file = h5py.File(filepath)
        self.feature_names = features
        self.coordinate_names = coordinates
        self.graph_feature_names = None

        # Create class targets
        interaction_type = np.array(self.file['InteractionType'], dtype=np.int8)
        pdg_encoding = np.array(self.file['PDGEncoding'], dtype=np.int8)
        has_track = np.logical_and(np.abs(pdg_encoding) == 14, interaction_type == 1)
        targets = has_track.astype(np.int)

        number_vertices = np.array(self.file['NumberVertices'])
        event_offsets = number_vertices.cumsum() - number_vertices

        # Apply filters to the dataset
        filter = event_filter(self.file, min_track_length=min_track_length, max_cascade_energy=max_cascade_energy)
        if balance_dataset:
            # Initialize numpys random seed with a hash of the filepath such that always the same indices get chosen 'randomly'
            np.random.seed(int(hashlib.sha1(filepath.encode()).hexdigest(), 16) & 0xFFFFFFFF)
            classes, class_counts = np.unique(targets[filter], return_counts=True)
            min_class_size = np.min(class_counts)
            for class_ in classes:
                # Remove the last samples of the larger class
                class_idx = np.where(np.logical_and((targets == class_), filter))[0]
                np.random.shuffle(class_idx)
                filter[class_idx[min_class_size : ]] = False
            idxs = np.where(filter)[0]
            # Assert that the class counts are the same now
            _, class_counts = np.unique(targets[idxs], return_counts=True)
            assert np.allclose(class_counts, min_class_size)
            #print(class_counts, idxs.shape)
            print(f'Reduced dataset to {min_class_size} samples per class ({idxs.shape[0]} / {targets.shape[0]})')
        else:
            idxs = np.arange(targets.shape[0])
        self.number_vertices = number_vertices[idxs]
        self.event_offsets = self.number_vertices.cumsum() - self.number_vertices
        self.targets = targets[idxs]

        # Create memmaps for features and coordinates for faster access during training
        os.makedirs(os.path.dirname(memmap_directory), exist_ok=True)

        # Load precomputed memmaps based on the hash of the columns, filename and index set
        idxs_hash = hashlib.sha1(idxs.data).hexdigest()
        features_hash = hashlib.sha1(str([
            [os.path.relpath(filepath)] + features
        ]).encode()).hexdigest()
        coordinates_hash = hashlib.sha1(str([
            [os.path.relpath(filepath)] + coordinates
        ]).encode()).hexdigest()

        feature_memmap_path = os.path.join(memmap_directory, f'hd5_features_{features_hash}_{idxs_hash}')
        coordinate_memmap_path = os.path.join(memmap_directory, f'hd5_coordinates_{coordinates_hash}_{idxs_hash}')

        if not os.path.exists(feature_memmap_path) or not os.path.exists(coordinate_memmap_path):
            # Create an index set that operates on vertex features, which is used to build memmaps efficiently
            vertex_idxs = np.concatenate([np.arange(start, end) for start, end in zip(self.event_offsets, self.event_offsets + self.number_vertices)]).tolist()
        
        if not os.path.exists(feature_memmap_path):
            # Create a new memmap for all features
            self.features = np.memmap(feature_memmap_path, shape=(self.number_vertices.sum(), len(self.feature_names)), dtype=np.float64, mode='w+')
            for feature_idx, feature in enumerate(self.feature_names):
                print(f'\rCreating column for feature {feature}', end='\r')
                self.features[:, feature_idx] = np.array(self.file.get(feature))[vertex_idxs]
            print(f'\nCreated feature memmap {feature_memmap_path}')
        else:
            self.features = np.memmap(feature_memmap_path, shape=(self.number_vertices.sum(), len(self.feature_names)), dtype=np.float64)

        if not os.path.exists(coordinate_memmap_path):
            # Create a new memmap for all features
            self.coordinates = np.memmap(coordinate_memmap_path, shape=(self.number_vertices.sum(), len(self.coordinate_names)), dtype=np.float64, mode='w+')
            for coordinate_idx, coordinate in enumerate(self.coordinate_names):
                print(f'\rCreating column for coordinate {coordinate}', end='\r')
                self.coordinates[:, coordinate_idx] = np.array(self.file.get(coordinate))[vertex_idxs]
            print(f'\nCreated coordinate memmap {coordinate_memmap_path}')
        else:
            self.coordinates = np.memmap(coordinate_memmap_path, shape=(self.number_vertices.sum(), len(self.coordinate_names)), dtype=np.float64)

        # Sanity checks
        endpoints = self.number_vertices + self.event_offsets
        assert(np.max(endpoints)) <= self.features.shape[0]

    def __len__(self):
        return self.number_vertices.shape[0]
    
    def __getitem__(self, idx):
        N = self.number_vertices[idx]
        offset = self.event_offsets[idx]
        X = self.features[offset : offset + N]
        C = self.coordinates[offset : offset + N]
        return X, C, self.targets[idx]

    @staticmethod
    def collate(samples):
        """ Collator for the dataset wrapper.
        
        Parameters:
        -----------
        samples : list
            A list of tuples representing the different inputs and targets.
        
        Returns:
        --------
        X : torch.FloatTensor, shape [batch_size, N, D]
            The vertex features.
        C : torch.FloatTensor, shape [batch_size, N, num_coordinates]
            Pariwise distances.
        masks : torch.FloatTensor, shape [batch_size, N, N]
            Adjacency matrix masks.
        targets : torch.FloatTensor, shape [batch_size, N, 1]
            Class labels.
        """
        X = [sample[0] for sample in samples]
        C = [sample[1] for sample in samples]
        y = [sample[2] for sample in samples]

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
        return features, coordinates, masks, targets

def event_filter(file, min_track_length=None, max_cascade_energy=None):
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

    Returns:
    --------
    filter : ndarray, shape [N], dtype=np.bool
        Only events that passed all filters are masked with True.
    """
    track_length = np.array(file['TrackLength'])
    cascade_energy = np.array(file['CascadeEnergy'])
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
    
    return filter









