import numpy as np
import pickle
import tables

__all__ = ['TestDataset',]

#test_data = '../test_data/test_data.pkl'
test_data = '../test_data/data_centered_reco.pkl'

class TestDataset(object):
    """ Dataset mock from a single pickle for testing the model. """

    def __init__(self, pickle_path = test_data, validation_portion=0.2, shuffle=True, 
        interaction_types = (b'numu', b'nue', b'nutau')):
        """ Initializes the test data.
        
        Parameters:
        -----------
        pickle_path : str
            Path to the pickle file containing all three interaction types.
        validation_portion : float
            The fraction of the dataset to be used for validation during training.
        shuffle : bool
            If True, the indices will be shuffled randomly.
        interaction_types : iterable
            All interaction types to be considered.
        """
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        self.features, self.coordinates, self.targets, self.baselines = [], [], [], []
        for class_label, interaction_type in enumerate(interaction_types):
            has_track = interaction_type == b'numu'
            self.features += data[interaction_type][b'features']
            self.coordinates += data[interaction_type][b'coordinates']
            self.targets += [np.float32(has_track)] * len(data[interaction_type][b'features'])
            for baseline in data[interaction_type][b'baselines']:
                assert (baseline[0] == baseline).all()
                self.baselines.append(baseline[0])
        self.features = np.array(self.features)
        self.coordinates = np.array(self.coordinates)
        self.targets = np.array(self.targets)
        self.baselines = np.array(self.baselines)
        idx = np.arange(len(self.features))
        validation_start = int(len(self.features) * (1 - validation_portion))
        if shuffle:
            np.random.shuffle(idx)
        self.idx_train = idx[:validation_start]
        self.idx_validation = idx[validation_start:]

    def get_baseline_accuracy(self, threshold=2.0):
        """ Calculates the accuracy on the validation set using the baseline method. 
        
        Parameters:
        -----------
        threshold : float
            The llh delta value to treshold the classification. All events greater or equal than
            the threshold will be assigned the track class.
        
        Returns:
        --------
        accuracy : float
            The baseline accuracy on the validation data.
        """
        y_true = self.targets[self.idx_validation]
        y_baseline = (self.baselines[self.idx_validation] >= 2.0).astype(np.float)
        return (y_true == y_baseline).sum() / y_true.shape[0]

    def get_class_prior(self):
        """ Returns the class prior, i.e. the number of samples per class for the dataset. 
        
        Returns:
        --------
        class_prior : dict
            A dict mapping from class label to float fractions.
        """
        labels, counts = np.unique(self.targets[self.idx_train], return_counts=True)
        class_prior = {}
        for label, count in zip(labels, counts):
            class_prior[label] = count / self.idx_train.shape[0]
        return class_prior

    def size(self, train=True):
        return len(self.idx_train) if train else len(self.idx_validation)

    def get_batches(self, batch_size=32, train=True):
        """ Generator method for retrieving the data. 
        Parameters:
        -----------
        batch_size : int
            The batch size.
        train : bool
            If true, returns training data, otherwise validation data.
        
        Yields:
        -------
        features_padded : ndarray, shape [batch_size, N_max, D]
            Feature matrices for n graphs.
        coordinates_padded : ndarray, shape [batch_size, N_max, 3]
            Coordinate matrices for n graphs.
        masks : ndarray, shape [batch_size, N_max, N_max]
            Adjacency matrix mask for each of the graphs.
        targets : ndarray, shape [batch_size]
            Class labels.
        """
        idxs = self.idx_train if train else self.idx_validation
        # Loop over the dataset
        while True:
            for idx in range(0, len(self.idx_train), batch_size):
                batch_idxs = idxs[idx : min(len(self.idx_train), idx + batch_size)]
                features, coordinates, masks = self.pad_batch(batch_idxs)
                targets = self.targets[batch_idxs]
                yield [features, coordinates, masks], targets
                

    def pad_batch(self, batch_idxs):
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
        num_features = 6 #features[0].shape[1]
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

