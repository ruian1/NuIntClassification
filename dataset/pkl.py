import numpy as np
import pickle
import tables
from .dataset import Dataset

#test_data = '../test_data/test_data.pkl'
test_data = '../test_data/data_centered_reco.pkl'

class PickleDataset(Dataset):
    """ Dataset from a single pickle. """

    def __init__(self, path = test_data, validation_portion=0.1, test_portion=0.1, shuffle=True, 
        interaction_types = (b'numu', b'nue', b'nutau')):
        """ Initializes the test data.
        
        Parameters:
        -----------
        path : str
            Path to the pickle file containing all three interaction types.
        validation_portion : float
            The fraction of the dataset to be used for validation during training.
        test_portion : float
            The fraction of the dataset to be used for testing only after training.
        shuffle : bool
            If True, the indices will be shuffled randomly.
        interaction_types : iterable
            All interaction types to be considered.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        self.features, self.coordinates, self.targets, self.delta_loglikelihood = [], [], [], []
        for class_label, interaction_type in enumerate(interaction_types):
            has_track = interaction_type == b'numu'
            self.features += data[interaction_type][b'features']
            self.coordinates += data[interaction_type][b'coordinates']
            self.targets += [np.float32(has_track)] * len(data[interaction_type][b'features'])
            for delta_llh in data[interaction_type][b'baselines']:
                assert (delta_llh[0] == delta_llh).all()
                self.delta_loglikelihood.append(delta_llh[0])
        self.features = np.array(self.features)
        self.coordinates = np.array(self.coordinates)
        self.targets = np.array(self.targets)
        self.delta_loglikelihood = np.array(self.delta_loglikelihood)
        idx = np.arange(len(self.features))
        validation_start = int(len(self.features) * validation_portion)
        training_start = int(len(self.features) * (validation_portion + test_portion))
        if shuffle:
            np.random.shuffle(idx)
        self.idx_test = idx[ : validation_start]
        self.idx_val = idx[validation_start : training_start]
        self.idx_train = idx[training_start : ]

    def get_number_features(self):
        """ Returns the number of features in the dataset. 
        
        Returns:
        --------
        num_features : int
            The number of features the input graphs have.
        """
        return self.features[0].shape[1]

    def get_padded_batch(self, batch_idxs):
        """ Pads a batch with zeros and creats a mask.
        
        Parameters:
        -----------
        batch_idxs : ndarray, shape [batch_size]
            The indices of the batch.
        
        Returns:
        --------
        features_padded : ndarray, shape [batch_size, N_max, D]
            The feature matrices in the batch.
        coordinates_padded : ndarray, shape [batch_size, N_max, 3]
            The coordiante matrices for graphs in the batch.
        masks : ndarray, shape [batch_size, N_max, N_max]
            Adjacency matrix masks for each graph in the batch.
        """
        features = self.features[batch_idxs]
        coordinates = self.coordinates[batch_idxs]
        batch_size = len(batch_idxs)
        padded_size = max(map(lambda x: x.shape[0], self.features[batch_idxs]))
        num_features = features[0].shape[1]
        num_coordinates = coordinates[0].shape[1]
        features_padded = np.zeros((batch_size, padded_size, num_features))
        coordinates_padded = np.zeros((batch_size, padded_size, num_coordinates))
        masks = np.zeros((batch_size, padded_size, padded_size))
        for idx, (f, c) in enumerate(zip(features, coordinates)):
            # For testing remove features 3, 4, 5, 6, 7
            # f = np.delete(f, [3, 4, 5, 6, 7], axis=1)
            features_padded[idx, :f.shape[0], :] = f
            coordinates_padded[idx, :c.shape[0], :] = c
            masks[idx, :f.shape[0], :f.shape[0]] = 1
        return features_padded, coordinates_padded, masks

