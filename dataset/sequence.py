from .hd5 import *

class ShuffledSequenceTorchHD5Dataset(ShuffledTorchHD5Dataset):
    """ Class to represent a pre-shuffled PyTorch dataset containing a sequence of DOM hits."""

    def __init__(self, filepath, features=[], balance_dataset=False, min_track_length=None, max_cascade_energy=None, flavors=None, currents=None,
        memmap_directory='./memmaps', close_file=True, 
        class_weights=None):
        """ Initializes the Dataset. 
        
        Parameters:
        -----------
        filepath : str
            A path to the hd5 file.
        features : list
            A list of dataset columns in the HD5 File that represent the DOM features.
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
        super().__init__(filepath)
        self.feature_names = features

        targets = self._create_targets()
        self._idxs = self._create_idxs(targets, balance_dataset, min_track_length, max_cascade_energy, flavors, currents)
        number_hits = np.array(self.file['NumberVertices'])
        sample_offsets = (number_hits.cumsum() - number_hits)[self._idxs]
        self.number_hits = number_hits[self._idxs]
        self.sample_offsets = (number_hits.cumsum() - number_hits)[self._idxs] # 'self.sample_offsets' refers to the feature matrix, 'sample_offsets' to the hd5 file

        # Create memmaps for features
        os.makedirs(os.path.dirname(memmap_directory), exist_ok=True)

        # Load precomputed memmaps based on the hash of the columns
        self._idxs_hash = hashlib.sha1(self._idxs.data).hexdigest()
        features_hash = hashlib.sha1(str([
            [os.path.relpath(filepath)] + features
        ]).encode()).hexdigest()
        feature_memmap_path = os.path.join(memmap_directory, f'hd5_features_{features_hash}_{self._idxs_hash}')
        
        if not os.path.exists(feature_memmap_path):
            # Create an index set that operates on vertex features, which is used to build memmaps efficiently
            _event_idxs = np.concatenate([np.arange(start, end) for start, end in zip(sample_offsets, sample_offsets + self.number_hits)]).tolist()
            number_samples = len(_event_idxs)
        else:
            _event_idxs = None
            number_samples = int(self.number_hits.sum())

        self.features = self._create_feature_memmap(feature_memmap_path, _event_idxs, self.feature_names, number_samples=number_samples)
        self.weights = self._compute_weights(class_weights, targets)
        
         # Sanity checks
        endpoints = self.number_hits + self.sample_offsets
        assert(np.max(endpoints)) <= self.features.shape[0]
       
        if close_file:
            self.file.close()
            self.file = filepath

    def __len__(self):
        return self.number_hits.shape[0]
    
    def __getitem__(self, idx):
        N = self.number_hits[idx]
        offset = self.sample_offsets[idx]
        X = self.features[offset : offset + N]
        return X, self.targets[idx], self.weights[idx]

    def get_maximal_sequence_length(self):
        """ Returns the length of the maximal sequence. 
        
        Returns:
        --------
        max_sequence_length : int
            The maximal sequence length.
        """
        return self.number_hits.max()

    @staticmethod
    def collate(samples):
        """ Collator for the dataset wrapper.
        
        Parameters:
        -----------
        samples : list
            A list of tuples representing the different inputs and targets.
        
        Returns:
        --------
        X : torch.FloatTensor, shape [seq_length, batch_size, D]
            Features for the DOM hits.
        outputs : torch.FloatTensor, shape [batch_size, 1]
            Class labels. 
        weights : torch.FloatTensor, shape [batch_size, 1] or None
            Weights for each sample.
        """
        X, y, w = zip(*samples)
    
        # Pad to the longest sequence size
        batch_size = len(X)
        sequence_length = max(map(lambda features: features.shape[0], X))
        features = np.zeros((sequence_length, batch_size, X[0].shape[1]))        
        for idx, X_i in enumerate(X):
            features[ : X_i.shape[0], idx] = X_i
        
        # Make torch tensors
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        features = torch.FloatTensor(features).to(device)
        targets = torch.FloatTensor(y).to(device).unsqueeze(1)
        weights = torch.FloatTensor(w).to(device).unsqueeze(1)
        return features, targets, weights