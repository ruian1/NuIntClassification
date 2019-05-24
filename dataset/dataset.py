import numpy as np

class Dataset(object):
    """ Parent class for different dataset implementations. """

    def get_baseline_accuracy(self, dataset='val', threshold=2.0):
        """ Calculates the accuracy on the validation set using the baseline method. 
    
        Parameters:
        -----------
        dataset : 'train' or 'val' or 'test'
            The dataset to access.
        threshold : float
            The llh delta value to treshold the classification. All events greater or equal than
            the threshold will be assigned the track class.
        
        Returns:
        --------
        accuracy : float
            The baseline accuracy on the validation data.
        """
        idx = self._get_idx(dataset)
        y_true = self.targets[idx]
        y_baseline = (self.delta_loglikelihood[idx] >= threshold).astype(np.float)
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
    
    def size(self, dataset='train'):
        """ Gets the number of samples in a dataset. 
        
        Parameters:
        -----------
        dataset : 'train' or 'val' or 'test'
            The dataset to access.
        
        Returns:
        --------
        size : int
            The size of the dataset.
        """
        idx = self._get_idx(dataset)
        return len(idx)

    def get_number_features(self):
        """ Returns the number of features in the dataset. 
        
        Returns:
        --------
        num_features : int
            The number of features the input graphs have.
        """
        raise NotImplementedError

    def _get_idx(self, dataset):
        """ Returns all indices associated with a certain dataset type.
        
        Parameters:
        -----------
        dataset : 'train' or 'val' or 'test'
            The dataset to access. 
            
        Returns:
        --------
        idx : ndarray, shape [N]
            The indices for the respective dataset.    
        """
        if dataset == 'train':
            return self.idx_train
        elif dataset == 'val':
            return self.idx_val
        elif dataset == 'test':
            return self.idx_test
        else:
            raise RuntimeError(f'Unkown dataset type {dataset}')

    def get_batches(self, batch_size=32, dataset='train'):
        """ Generator method for retrieving the data. 
        Parameters:
        -----------
        batch_size : int
            The batch size.
        dataset : 'train' or 'val' or 'test'
            The dataset to access.
        
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
        idxs = self._get_idx(dataset)
        # Loop over the dataset
        while True:
            for idx in range(0, idxs.shape[0], batch_size):
                batch_idxs = idxs[idx : min(len(self.idx_train), idx + batch_size)]
                features, coordinates, masks = self.get_padded_batch(batch_idxs)
                targets = self.targets[batch_idxs]
                yield [features, coordinates, masks], targets