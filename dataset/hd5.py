import numpy as np
import pickle
import tables
from collections import defaultdict

__all__ = ['HD5Dataset',]

class HD5Dataset(object):
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
        # Open all file handles
        self.files = {}
        for label in files:
            self.files[label] = [tables.open_file(filepath) for filepath in files[label]]
        # Count the total number of samples in all files and create lookup array for the vertex subsets
        self.vertex_offsets = defaultdict(list)

        number_samples = 0
        for label in self.files:
            for table_file in self.files[label]:
                number_samples += len(table_file.root.NumberVertices.cols.value)
                number_vertices = np.array(table_file.root.NumberVertices.cols.value, dtype=np.int64)
                self.vertex_offsets[label].append(np.cumsum(number_vertices)- number_vertices)
        idx = np.arange(number_samples)
        if shuffle:
            np.random.shuffle(idx)
        first_validation_idx = int(test_portion * number_samples)
        first_training_idx = int((test_portion + validation_portion) * number_samples)
        self.idx_test = idx[ : first_validation_idx]
        self.idx_val = idx[first_validation_idx : first_training_idx]
        self.idx_test = idx[first_training_idx : ]

    def _get_sample(self, idx):
        """ Returns a single data sample.
        
        Parameters:
        -----------
        idx : int
            The data sample to obtain, possibly from """

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

    

    def __del__(self):
        # Close all open hd5 files
        for label in self.files:
            for file in self.files[label]:
                file.close()

if __name__ == '__main__':
    HD5Dataset({
        0 : [
            '../../data/data_dragon_3y_nue.hd5', '../../data/data_dragon_3y_nutau.hd5'
        ],
        1 : ['../../data/data_dragon_3y_numu.hd5']
    })




