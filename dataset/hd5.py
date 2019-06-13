import numpy as np
import pickle
import h5py
from collections import defaultdict
from .dataset import Dataset
import tempfile
import os
import hashlib

__all__ = ['HD5Dataset', 'RecurrentHD5Dataset']


def load_chunked(hd5file, column, memmap, chunksize, column_idx=None):
    """ Loads a column from a hd5file into a memmep chunkwise. 
    
    Paramters:
    ----------
    hd5file : hd5py.File
        The hd5file to load from.
    column : str
        The column to load.
    memmap : np.memmap, shape [N] or [N, D]
        The memory map to load the data into.
    chunksize : int
        The number of rows to transfer at once.
    column_idx : int or None
        The column in which to insert the data.
    """
    column = hd5file.get(column)
    steps = int(np.ceil(column.shape[0] / chunksize))
    for step in range(steps):
        idx_from, idx_to = step * chunksize, min((step + 1) * chunksize, column.shape[0])
        print(f'\r{step} of {steps}, slice {idx_from}:{idx_to}', end='\r')
        print('\n')
        if column_idx is not None:
            memmap[idx_from : idx_to, column_idx] = column[idx_from : idx_to]['item']
        else:
            memmap[idx_from : idx_to] = column[idx_from : idx_to]['item']
        

class HD5Dataset(Dataset):
    """ Class to iterate over an HDF5 Dataset. """

    def __init__(self, filepath, validation_portion=0.1, test_portion=0.1, shuffle=True, 
        features=['CumulativeCharge', 'Time', 'FirstCharge'], seed = None, max_chunk_size=50000000, balance_dataset=False,
        min_track_length=None, max_cascade_energy=None, memmap_directory='./memmaps'):
        """ Initlaizes the dataset wrapper from multiple hdf5 files, each corresponding to exactly one class label.
        
        Parameters:
        -----------
        filepath : str
            Path to the hdf5 datafile.
        validation_portion : float
            The fraction of the dataset that is kept from the model during training for validation.
        test_portion = 0.1
            The fraction of the dataset that is kept form the model during training and only evaulated after training has finished.
        shuffle : bool
            If True, the data will be shuffled randomly.
        features : list
            A list of feature columns that must be present as children of the root of the hdf5 file.
        seed : None or int  
            The seed of the numpy shuffling if given.
        max_cunk_size : int
            The number of rows that are transfered into memory maps at once.
        balance_dataset : bool
            If the dataset should be balanced such that each class contains the same number of samples.
        min_track_length : float or None
            Minimal track length all track-events must have.
        max_cascade_energy : float or None
            The maximal cascade energy all track events are allowed to have.
        memmap_directory : str
            Directory for memmaps.
        """
        super().__init__()
        filepath = os.path.relpath(filepath)
        self.file = h5py.File(filepath, 'r')
        self.feature_names = features

        self._create_targets()
        filters = event_filter(self.file, min_track_length=min_track_length, max_cascade_energy=max_cascade_energy)
        self._create_idx(validation_portion, test_portion, filters=filters, shuffle=shuffle, seed=seed, balanced=balance_dataset)

        # Create lookup arrays for the graph size of each sample and their offsets in the feature matrix 
        # since all the features of all graphs are stacked in one big table
        self.number_vertices = np.array(self.file['NumberVertices']['value'], dtype = np.int32)
        try:
            self.sample_offsets = np.array(self.file['Offset']['value'], dtype=np.int32)
        except:
            self.sample_offsets = np.cumsum(self.number_vertices) - self.number_vertices
        self.distances_offsets = self.number_vertices ** 2
        self.distances_offsets = np.cumsum(self.distances_offsets, dtype=np.int64) - self.distances_offsets

        memmap_hash = hashlib.sha1(str(self.feature_names + [filepath]).encode()).hexdigest()
        print(f'Created sha1 hash for features and data file {memmap_hash}')

        # Create memmaps in order to access the data without iterating and shuffling in place
        feature_memmap = os.path.join(memmap_directory, f'hd5_features_{memmap_hash}')
        if os.path.exists(feature_memmap):
            self.features = np.memmap(feature_memmap, shape=(int(self.number_vertices.sum()), len(self.feature_names)), dtype=np.float64)
            print(f'Loaded feature memmap {feature_memmap}.')
        else:
            self.features = np.memmap(feature_memmap, shape=(int(self.number_vertices.sum()), len(self.feature_names)), mode='w+', dtype=np.float64)
            for feature_idx, feature in enumerate(self.feature_names):
                if self.file.get(feature).shape[0] <= max_chunk_size:
                    self.features[:, feature_idx] = self.file.get(feature)['item']
                else:
                    load_chunked(self.file, feature, self.features, max_chunk_size, column_idx=feature_idx)
                print(f'Loaded feature {feature}')
            print(f'Created feature memmap {feature_memmap}')
        distances_memmap = os.path.join(memmap_directory, f'hd5_distances_{memmap_hash}', )
        if os.path.exists(distances_memmap):
            self.distances = np.memmap(distances_memmap, shape=self.file['Distances'].shape, dtype=np.float64)
            print(f'Created distances memmap {distances_memmap}.')
        else:
            self.distances = np.memmap(distances_memmap, shape=self.file['Distances'].shape, mode='w+', dtype=np.float64)
            if self.file['Distances'].shape[0] <= max_chunk_size:
                self.distances[:] = self.file['Distances']['item']
            else:
                load_chunked(self.file, 'Distances', self.distances, max_chunk_size, column_idx=None)
            print(f'Created distances memmap {distances_memmap}.')
        self.delta_loglikelihood = np.array(self.file.get('DeltaLLH')['value'])

    def _create_targets(self):
        """ Builds the targets for classification. """
        interaction_type = np.array(self.file['InteractionType']['value'], dtype=np.int8)
        pdg_encoding = np.array(self.file['PDGEncoding']['value'], dtype=np.int8)
        has_track = np.logical_and(np.abs(pdg_encoding) == 14, interaction_type == 1)
        self.targets = has_track.astype(np.int)

    def get_number_features(self):
        """ Returns the number of features in the dataset. 
        
        Returns:
        --------
        num_features : int
            The number of features the input graphs have.
        """
        return len(self.feature_names)

    def get_padded_batch(self, batch_idxs):
        """ Pads a batch with zeros and creats a mask.
        
        Parameters:
        -----------
        batch_idxs : ndarray, shape [batch_size]
            The indices of the batch.
        
        Returns:
        --------
        features : ndarray, shape [batch_size, N_max, D]
            The feature matrices in the batch.
        distances : ndarray, shape [batch_size, N_max, N_max]
            Precomputed pairwise distances for each graph.
        masks : ndarray, shape [batch_size, N_max, N_max]
            Adjacency matrix masks for each graph in the batch.
        """
        # Collect the features
        padded_number_vertices = np.max(self.number_vertices[batch_idxs])
        features = np.zeros((batch_idxs.shape[0], padded_number_vertices, len(self.feature_names)))
        distances = np.zeros((batch_idxs.shape[0], padded_number_vertices, padded_number_vertices))
        masks = np.zeros((batch_idxs.shape[0], padded_number_vertices, padded_number_vertices))
        for idx, batch_idx in enumerate(batch_idxs):
            number_vertices = self.number_vertices[batch_idx]
            offset = self.sample_offsets[batch_idx]
            distances_offset = self.distances_offsets[batch_idx]
            features[idx, : number_vertices, :] = self.features[offset : offset + number_vertices]
            distances[idx, : number_vertices, : number_vertices] = \
                self.distances[distances_offset : distances_offset + number_vertices ** 2].reshape((number_vertices, number_vertices))
            masks[idx, : number_vertices, : number_vertices] = 1
        return features, distances, masks


class RecurrentHD5Dataset(Dataset):
    """ Class to iterate over an HDF5 Dataset. """

    def __init__(self, filepath, validation_portion=0.1, test_portion=0.1, shuffle=True, 
        features=['CumulativeCharge', 'Time', 'FirstCharge'], max_chunk_size=100000000, balance_dataset=False,
        min_track_length=None, max_cascade_energy=None):
        """ Initlaizes the dataset wrapper from multiple hdf5 files, each corresponding to exactly one class label.
        
        Parameters:
        -----------
        filepath : str
            Path to the hdf5 datafile.
        validation_portion : float
            The fraction of the dataset that is kept from the model during training for validation.
        test_portion = 0.1
            The fraction of the dataset that is kept form the model during training and only evaulated after training has finished.
        shuffle : bool
            If True, the data will be shuffled randomly.
        features : list
            A list of feature columns that must be present as children of the root of the hdf5 file.
        max_cunk_size : int
            The number of rows that are transfered into memory maps at once.
        balance_dataset : bool
            If the dataset should be blanced such that each class contains the same number of samples.
        min_track_length : float or None
            Minimal track length all track-events must have.
        max_cascade_energy : float or None
            The maximal cascade energy all track events are allowed to have.
        """
        super().__init__()
        self.file = h5py.File(filepath, 'r')
        self.feature_names = features

        self._create_targets()
        filters = event_filter(self.file, min_track_length=min_track_length, max_cascade_energy=max_cascade_energy)
        self._create_idx(validation_portion, test_portion, filters=filters, shuffle=shuffle, seed=seed, balanced=balance_dataset)

        # Create lookup arrays for the graph size of each sample and their offsets in the feature matrix 
        # since all the features of all graphs are stacked in one big table
        self.number_vertices = np.array(self.file['NumberVertices']['value'], dtype = np.int64)
        self.number_steps = np.array(self.file['NumberSteps']['value'], dtype = np.int64)

        # Calculate the offsets to rebuild matrices from flattened feature vectors
        number_rows = self.number_vertices * self.number_steps
        self.feature_offsets = np.cumsum(number_rows, dtype=np.int64) - number_rows
        self.active_vertex = np.array(self.file['ActiveVertex']['item'], dtype=np.int)
        self.active_vertex_offsets = np.cumsum(self.number_steps) - self.number_steps
        self.distances_offsets = self.number_vertices ** 2
        self.distances_offsets = np.cumsum(self.distances_offsets, dtype=np.int64) - self.distances_offsets

        # Create memmaps in order to access the data without iterating and shuffling in place
        feature_file = tempfile.NamedTemporaryFile('w+')
        self.features = np.memmap(feature_file.name, shape=(int((self.number_vertices * self.number_steps).sum()), len(self.feature_names)))
        for feature_idx, feature in enumerate(self.feature_names):
            load_chunked(self.file, feature, self.features[:, feature_idx], max_chunk_size)
            print(f'Loaded feature {feature}')
        distances_file = tempfile.NamedTemporaryFile('w+')
        self.distances = np.memmap(distances_file.name, shape=self.file['Distances'].shape)
        load_chunked(self.file, 'Distances', self.distances, max_chunk_size)
        print('Created memory map arrays.')
        self.delta_loglikelihood = np.array(self.file.get('DeltaLLH')['value'])

    def _create_targets(self):
        """ Builds the targets for classification. """
        interaction_type = np.array(self.file['InteractionType']['value'], dtype=np.int8)
        pdg_encoding = np.array(self.file['PDGEncoding']['value'], dtype=np.int8)
        has_track = np.logical_and(np.abs(pdg_encoding) == 14, interaction_type == 1)
        self.targets = has_track.astype(np.int)

    def get_number_features(self):
        """ Returns the number of features in the dataset. 
        
        Returns:
        --------
        num_features : int
            The number of features the input graphs have.
        """
        return len(self.feature_names)

    def get_padded_batch(self, batch_idxs):
        """ Pads a batch with zeros and creats a mask.
        
        Parameters:
        -----------
        batch_idxs : ndarray, shape [batch_size]
            The indices of the batch.
        
        Returns:
        --------
        features : ndarray, shape [batch_size, num_steps, N_max, D]
            The feature matrices in the batch.
        distances : ndarray, shape [batch_size, num_steps, N_max, N_max]
            Pairwise distances.
        masks : ndarray, shape [batch_size, num_steps, N_max, N_max]
            Adjacency matrix masks for each graph in the batch.
        """
        # Calculate batch dimensions (with padding)
        padded_number_vertices = np.max(self.number_vertices[batch_idxs])
        padded_number_steps = np.max(self.number_steps[batch_idxs])

        features = np.zeros((batch_idxs.shape[0], padded_number_steps, padded_number_vertices, len(self.feature_names)))
        distances = np.zeros((batch_idxs.shape[0], padded_number_steps, padded_number_vertices, padded_number_vertices))
        masks = np.zeros((batch_idxs.shape[0], padded_number_steps, padded_number_vertices, padded_number_vertices))
        for idx, batch_idx in enumerate(batch_idxs):
            number_vertices = self.number_vertices[batch_idx]
            number_steps = self.number_steps[batch_idx]
            offset = self.feature_offsets[batch_idx]
            distances_offset = self.distances_offsets[batch_idx]
            features[idx, : number_steps, : number_vertices, :] = self.features[offset : offset + (number_vertices * number_steps)].reshape(
                (number_steps, number_vertices, -1))
            distances[idx, : number_steps, : number_vertices, : number_vertices] = \
                self.distances[distances_offset : distances_offset + (number_vertices ** 2)].reshape((number_vertices, number_vertices))
            for step in range(number_steps):
                largest_active_vertex_idx = self.active_vertex[self.active_vertex_offsets[batch_idx]]
                masks[idx, : step, : largest_active_vertex_idx, : largest_active_vertex_idx] = 1
        return features, distances, masks

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
    track_length = np.array(file['TrackLength']['value'])
    cascade_energy = np.array(file['CascadeEnergy']['value'])
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




if __name__ == '__main__':
    HD5Dataset('../data/data_dragon_sequential.hd5')









