import numpy as np
import pickle
import h5py
from collections import defaultdict
from .dataset import Dataset
import tempfile

__all__ = ['HD5Dataset', 'RecurrentHD5Dataset']

class HD5Dataset(Dataset):
    """ Class to iterate over an HDF5 Dataset. """

    def __init__(self, filepath, validation_portion=0.1, test_portion=0.1, shuffle=True, 
        features=['CumulativeCharge', 'Time', 'FirstCharge'], coordinates=['VertexX', 'VertexY', 'VertexZ'],
        seed = None):
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
        coordinates : list
            A list of columns that correspond to the spatial coordinates of the vertices.
        seed : None or int  
            The seed of the numpy shuffling if given.
        """
        super().__init__()
        self.file = h5py.File(filepath, 'r')
        self.feature_names = features
        self.coordinate_names = coordinates
        # Create lookup arrays for the graph size of each sample and their offsets in the feature matrix 
        # since all the features of all graphs are stacked in one big table
        self.number_vertices = np.array(self.file['NumberVertices']['value'], dtype = np.int32)
        try:
            self.sample_offsets = np.array(self.file['Offset']['value'], dtype=np.int32)
        except:
            self.sample_offsets = np.cumsum(self.number_vertices) - self.number_vertices
        idx = np.arange(self.number_vertices.shape[0])
        np.random.seed(seed)
        if shuffle:
            np.random.shuffle(idx)
        first_validation_idx = int(test_portion * idx.shape[0])
        first_training_idx = int((test_portion + validation_portion) * idx.shape[0])
        self.idx_test = idx[ : first_validation_idx]
        self.idx_val = idx[first_validation_idx : first_training_idx]
        self.idx_train = idx[first_training_idx : ]

        # Create memmaps in order to access the data without iterating and shuffling in place
        feature_file = tempfile.NamedTemporaryFile('w+')
        self.features = np.memmap(feature_file.name, shape=(int(self.number_vertices.sum()), len(self.feature_names)))
        for feature_idx, feature in enumerate(self.feature_names):
            self.features[:, feature_idx] = self.file.get(feature)['item']
            print(f'Loaded feature {feature}')
        coordinate_file = tempfile.NamedTemporaryFile('w+')
        self.coordinates = np.memmap(coordinate_file.name, shape=(int(self.number_vertices.sum()), len(self.coordinate_names)))
        for coordinate_idx, coordinate in enumerate(self.coordinate_names):
            self.coordinates[:, coordinate_idx] = self.file.get(coordinate)['item']
        print('Created memory map arrays.')
        self._create_targets()
        self.delta_loglikelihood = np.array(self.file.get('DeltaLLH')['value'])

    def _create_targets(self):
        """ Builds the targets for classification. """
        interaction_type = np.array(self.file['InteractionType']['value'], dtype=np.uint8)
        pdg_encoding = np.array(self.file['PDGEncoding']['value'], dtype=np.uint8)
        has_track = np.logical_and(pdg_encoding == 14, interaction_type == 1)
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
        coordinates : ndarray, shape [batch_size, N_max, 3]
            The coordiante matrices for graphs in the batch.
        masks : ndarray, shape [batch_size, N_max, N_max]
            Adjacency matrix masks for each graph in the batch.
        """
        # Collect the features
        padded_number_vertices = np.max(self.number_vertices[batch_idxs])
        features = np.zeros((batch_idxs.shape[0], padded_number_vertices, len(self.feature_names)))
        coordinates = np.zeros((batch_idxs.shape[0], padded_number_vertices, len(self.coordinate_names)))
        masks = np.zeros((batch_idxs.shape[0], padded_number_vertices, padded_number_vertices))
        for idx, batch_idx in enumerate(batch_idxs):
            number_vertices = self.number_vertices[batch_idx]
            offset = self.sample_offsets[batch_idx]
            features[idx, : number_vertices, :] = self.features[offset : offset + number_vertices]
            coordinates[idx, : number_vertices, :] = self.coordinates[offset : offset + number_vertices]
            masks[idx, : number_vertices, : number_vertices] = 1
        return features, coordinates, masks


class RecurrentHD5Dataset(Dataset):
    """ Class to iterate over an HDF5 Dataset. """

    def __init__(self, filepath, validation_portion=0.1, test_portion=0.1, shuffle=True, 
        features=['CumulativeCharge', 'Time', 'FirstCharge'], coordinates=['VertexX', 'VertexY', 'VertexZ']):
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
        coordinates : list
            A list of columns that correspond to the spatial coordinates of the vertices.
        """
        super().__init__()
        self.file = h5py.File(filepath, 'r')
        self.feature_names = features
        self.coordinate_names = coordinates
        # Create lookup arrays for the graph size of each sample and their offsets in the feature matrix 
        # since all the features of all graphs are stacked in one big table
        self.number_vertices = np.array(self.file['NumberVertices']['value'], dtype = np.int64)
        self.sample_offsets = np.array(self.file['Offset']['value'], dtype=np.int64)
        self.vertex_offsets = np.cumsum(self.number_vertices) - self.number_vertices
        self.number_steps = np.array(self.file['NumberSteps']['value'], dtype=np.int64)
        idx = np.arange(self.number_vertices.shape[0])
        if shuffle:
            np.random.shuffle(idx)
        first_validation_idx = int(test_portion * idx.shape[0])
        first_training_idx = int((test_portion + validation_portion) * idx.shape[0])
        self.idx_test = idx[ : first_validation_idx]
        self.idx_val = idx[first_validation_idx : first_training_idx]
        self.idx_train = idx[first_training_idx : ]

        # Create memmaps in order to access the data without iterating and shuffling in place
        feature_file = tempfile.NamedTemporaryFile('w+')
        self.features = np.memmap(feature_file.name, shape=(int((self.number_vertices * self.number_steps).sum()), len(self.feature_names)))
        print(self.features.shape)
        for feature_idx, feature in enumerate(self.feature_names):
            self.features[:, feature_idx] = self.file.get(feature)['item']
        coordinate_file = tempfile.NamedTemporaryFile('w+')
        self.coordinates = np.memmap(coordinate_file.name, shape=(int(self.number_vertices.sum()), len(self.coordinate_names)))
        for coordinate_idx, coordinate in enumerate(self.coordinate_names):
            self.coordinates[:, coordinate_idx] = self.file.get(coordinate)['item']
        print('Created memory map arrays.')
        self._create_targets()
        self.delta_loglikelihood = np.array(self.file.get('DeltaLLH')['value'])

    def _create_targets(self):
        """ Builds the targets for classification. """
        interaction_type = np.array(self.file['InteractionType']['value'], dtype=np.uint8)
        pdg_encoding = np.array(self.file['PDGEncoding']['value'], dtype=np.uint8)
        has_track = np.logical_and(pdg_encoding == 14, interaction_type == 1)
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
        coordinates : ndarray, shape [batch_size, num_steps, N_max, 3]
            The coordiante matrices for graphs in the batch.
        masks : ndarray, shape [batch_size, num_steps, N_max, N_max]
            Adjacency matrix masks for each graph in the batch.
        """
        # Collect the features
        padded_number_vertices = np.max(self.number_vertices[batch_idxs])
        padded_number_steps = np.max(self.number_steps[batch_idxs])

        features = np.zeros((batch_idxs.shape[0], padded_number_steps, padded_number_vertices, len(self.feature_names)))
        coordinates = np.zeros((batch_idxs.shape[0], padded_number_steps, padded_number_vertices, len(self.coordinate_names)))
        masks = np.zeros((batch_idxs.shape[0], padded_number_steps, padded_number_vertices, padded_number_vertices))
        for idx, batch_idx in enumerate(batch_idxs):
            number_vertices = self.number_vertices[batch_idx]
            number_steps = self.number_steps[batch_idx]
            offset = self.sample_offsets[batch_idx]
            vertex_offset = self.vertex_offsets[batch_idx]
            features[idx, : number_steps, : number_vertices, :] = self.features[offset : offset + (number_vertices * number_steps)].reshape(
                (number_steps, number_vertices, -1))
            coordinates[idx, : number_steps,  : number_vertices, :] = self.coordinates[vertex_offset : vertex_offset + number_vertices].reshape(
                (number_vertices, -1))
            masks[idx, : number_steps,  : number_vertices, : number_vertices] = 1
        return features, coordinates, masks

if __name__ == '__main__':
    HD5Dataset('../data/data_dragon_sequential.hd5')









