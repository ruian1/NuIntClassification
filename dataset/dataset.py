import numpy as np
from sklearn.utils import class_weight

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

    def get_class_weights(self):
        """ Returns a weighting to the classes.
        
        Returns:
        --------
        class_prior : dict
            A dict mapping from class label to float fractions.
        """
        labels, counts = np.unique(self.targets[self.idx_train], return_counts=True)
        weights = class_weight.compute_class_weight('balanced', labels, self.targets)
        class_prior = {}
        for label, weight in zip(labels, weights):
            class_prior[label] = weight
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

    def _create_idx(self, validation_portion, test_portion, filters=None, shuffle=True, seed=None, balanced=True, min_track_length=None):
        """ Creates indices for training, validation and testing. 
        
        Parameters:
        -----------
        validation_portion : float
            The portion of the dataset that is used for validation.
        test_portion : float
            The portion of the dataset that is used for testing.
        filters : ndarray, shape [N], dtype bool
            Only elements that are masked with true will be considered.
        shuffle : bool
            If the data should be shuffled.
        seed : object
            The seed for data shuffling.
        balanced : object
            If the dataset casses will be balanced.
        """
        N = self.targets.shape[0]
        if filters is None:
            filters = np.ones((N,), dtype=np.bool)

        np.random.seed(seed)
        if balanced:
            classes, class_counts = np.unique(self.targets[filters], return_counts=True)
            min_class_size = np.min(class_counts)
            for class_ in classes:
                # Get and shuffle the idx of data samples of this class
                class_idx = np.where(np.logical_and((self.targets == class_), filters))[0]
                if shuffle:
                    np.random.shuffle(class_idx)
                filters[class_idx[min_class_size : ]] = False
            idxs = np.where(filters)[0]
            # Assert that the class counts are the same now
            _, class_counts = np.unique(self.targets[idxs], return_counts=True)
            assert np.allclose(class_counts, min_class_size)
            print(f'Reduced dataset to {min_class_size} samples per class ({idxs.shape[0]} / {N})')
        else:
            idxs = np.where(filters)[0]
            if shuffle:
                np.random.shuffle(idxs)

        first_validation_idx = int(test_portion * idxs.shape[0])
        first_training_idx = int((test_portion + validation_portion) * idxs.shape[0])
        self.idx_test = idxs[ : first_validation_idx]
        self.idx_val = idxs[first_validation_idx : first_training_idx]
        self.idx_train = idxs[first_training_idx : ]
