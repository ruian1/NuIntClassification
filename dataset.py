import numpy as np
import pickle
import tables

test_data = '../test_data/test_data.pkl'

class TestDataset(object):
    """ Dataset mock from a single pickle for testing the model. """

    def __init__(self, pickle_path = '../test_data/test_data.pkl', validation_portion=0.2, shuffle=True, 
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
        self.features, self.coordinates, self.targets = [], [], []
        for class_label, interaction_type in enumerate(interaction_types):
            has_track = interaction_type == b'numu'
            self.features += data[interaction_type][b'features']
            self.coordinates += data[interaction_type][b'coordinates']
            self.targets += [np.float32(has_track)] * len(data[interaction_type][b'features'])
        self.features = np.array(self.features)
        self.coordinates = np.array(self.coordinates)
        self.targets = np.array(self.targets)
        idx = np.arange(len(self.features))
        validation_start = int(len(self.features) * (1 - validation_portion))
        if shuffle:
            np.random.shuffle(idx)
        self.idx_train = idx[:validation_start]
        self.idx_validation = idx[validation_start:]

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
        num_features = features[0].shape[1]
        num_coordinates = coordinates[0].shape[1]
        features_padded = np.zeros((batch_size, padded_size, num_features))
        coordinates_padded = np.zeros((batch_size, padded_size, num_coordinates))
        masks = np.zeros((batch_size, padded_size, padded_size))
        for idx, (f, c) in enumerate(zip(features, coordinates)):
            features_padded[idx, :f.shape[0], :] = f
            coordinates_padded[idx, :c.shape[0], :] = c
            masks[idx, :f.shape[0], :f.shape[0]] = 1
        return features_padded, coordinates_padded, masks


class HDF5Dataset(object):
    """ Class to iterate over an HDF5 Dataset. """

    def __init__(self, files, validation_portion=0.1, test_portion=0.1, shuffle=True):
        """ Initlaizes the dataset wrapper from multiple hdf5 files, each corresponding to exactly one class label.
        
        Parameters:
        -----------
        files : dict
            A dict that maps class labels (integers) to lists of hdf5 file paths.
        validation_portion : float
            The fraction of the dataset that is kept from the model during training for validation.
        test_portion = 0.1
            The fraction of the dataset that is kept form the model during training and only evaulated after training has finished.
        shuffle : bool
            If True, the data will be shuffled randomly.
        """
        self.files = {
            label : [tables.open_file(filepath) for filepath in files[label]] for label in files
        }
        # Count the total number of samples in all files
        number_samples = 0
        for label in self.files:
            for file in self.files[label]:
                number_samples += len(table_file.root.VertexOffsets.cols.item)
        self.idx = np.arange(self.number_samples)

