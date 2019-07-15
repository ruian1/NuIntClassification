import numpy as np
import pickle
import h5py
from collections import defaultdict
import tempfile
import os
import hashlib
from sklearn.metrics import pairwise_distances
from sklearn.utils.class_weight import compute_sample_weight

import torch.utils.data

class ShuffledTorchHD5Dataset(torch.utils.data.Dataset):
    """ Baseclass for all hd5 datasets that are preshuffled. """

    def __init__(self, filepath):
        """ Initializes the dataset. 
        
        Parameters:
        -----------
        filepath : str
            A path to the hd5 file.
        """
        self.file = h5py.File(filepath)

    def _create_targets(self):
        """ Creates targets for classification based on some filtered indices. 
        
        Returns:
        --------
        targets : ndarray, shape [N]
            Class labels.
        """
        interaction_type = np.array(self.file['InteractionType'], dtype=np.int8)
        pdg_encoding = np.array(self.file['PDGEncoding'], dtype=np.int8)
        has_track = np.logical_and(np.abs(pdg_encoding) == 14, interaction_type == 1)
        return has_track.astype(np.int)

    def _create_idxs(self, targets, balance_dataset, min_track_length, max_cascade_energy, flavors, currents):
        """ Creates indices for training by application of filters and balancing.

        Parameters:
        -----------
        targets : ndarray, shape [N]
            Class labels.
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


        Returns:
        --------
        idxs : ndarray, shape [N]
            Indices that passed the filter and balancing.
        """
        filter = event_filter(self.file, min_track_length=min_track_length, max_cascade_energy=max_cascade_energy, flavors=flavors, currents=currents)
        if balance_dataset:
            # Initialize numpys random seed with a hash of the filepath such that always the same indices get chosen 'randomly'
            np.random.seed(int(hashlib.sha1(self.file.filename.encode()).hexdigest(), 16) & 0xFFFFFFFF)
            classes, class_counts = np.unique(targets[filter], return_counts=True)
            print(f'Classes {classes}; Class counts {class_counts}')
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
            print(f'Reduced dataset to {min_class_size} samples per class ({idxs.shape[0]} / {targets.shape[0]})')
        else:
            idxs = np.where(filter)[0]
        return idxs

    def _create_feature_memmap(self, path, idxs, features, number_samples=None):
        """ Creates a memory map based on some feature columns of the hd5 file.
        
        Parameters:
        -----------
        path : str
            The path to the memory map.
        idxs : list or None
            All indices to get from the file.
        features : list
            All the feature columns to get.
        number_samples : int or None
            If given, the number of samples in the memmap.
            
        Returns:
        --------
        memmap : np.memmaps, shape [N, len(features)]
            The feature memmap.
        """
        if number_samples is None:
            number_samples = len(idxs)

        if not os.path.exists(path):
            # Create a new memmap
            memmap = np.memmap(path, shape=(number_samples, len(features)), dtype=np.float64, mode='w+')
            for feature_idx, feature in enumerate(features):
                print(f'\rCreating column for feature {feature}', end='\r')
                memmap[:, feature_idx] = np.array(self.file.get(feature))[idxs]
            print(f'\nCreated feature memmap {path}')
        else:
            memmap = np.memmap(path, shape=(number_samples, len(features)), dtype=np.float64)
        return memmap

    def _compute_weights(self, class_weights, targets):
        """ Computes the class weights for each sample.
        
        Parameters:
        -----------
        class_weights : str or dict or None
            A string to indicate the method of weighting or preset class weights for each class.
        targets : ndarray, shape [N]
            Class labels.
    
        Returns:
        --------
        weights : ndarray, shape [N]
            Weights per data sample.
        """
        # Compute weights, if a dictionary is given, string keys must be converted to int keys
        if isinstance(class_weights, dict):
            class_weights = {int(class_) : weight for class_, weight in class_weights.items()}
        return compute_sample_weight(class_weights, targets)

def event_filter(file, min_track_length=None, max_cascade_energy=None, min_total_energy=None, max_total_energy=None, 
    flavors=None, currents=None):
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
    min_total_energy : float or None
        All events with a total energy (cascade + muon) less than that will be excluded.
    max_total_energy : float or None
        All events with a total energy (cascade + muon) more than that will be excluded.
    flavors : list or None
        Only certain neutrino flavor events will be considered if given.
    currents : list or None
        Only certain current events will be considered if given. 

    Returns:
    --------
    filter : ndarray, shape [N], dtype=np.bool
        Only events that passed all filters are masked with True.
    """
    track_length = np.array(file['TrackLength'])
    cascade_energy = np.array(file['CascadeEnergy'])
    muon_energy = np.array(file['MuonEnergy'])
    muon_energy[np.isnan(muon_energy)] = 0
    total_energy = cascade_energy.copy()
    total_energy[np.isnan(total_energy)] = 0
    total_energy += muon_energy

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
    
    # Flavor filter
    if flavors is not None:
        pdg_encoding = np.array(file['PDGEncoding'])
        flavor_mask = np.zeros_like(filter, dtype=np.bool)
        for flavor in flavors:
            flavor_mask[np.abs(pdg_encoding) == flavor] = True
        filter = np.logical_and(filter, flavor_mask)
        print(f'After Flavor filter {filter.sum()} / {filter.shape[0]} events remain.')

    # Current filter
    if currents is not None:
        interaction_type = np.array(file['InteractionType'])
        current_mask = np.zeros_like(filter, dtype=np.bool)
        for current in currents:
            current_mask[np.abs(interaction_type) == current] = True
        filter = np.logical_and(filter, current_mask)
        print(f'After Current filter {filter.sum()} / {filter.shape[0]} events remain.')

    # Min total nergy filter
    if min_total_energy is not None:
        idx_removed = np.where(total_energy < min_total_energy)[0]
        filter[idx_removed] = False
        print(f'After Min Total Energy filter {filter.sum()} / {filter.shape[0]} events remain.')

    # Max total energy filter
    if max_total_energy is not None:
        idx_removed = np.where(total_energy > max_total_energy)[0]
        filter[idx_removed] = False
        print(f'After Max Total Energy filter {filter.sum()} / {filter.shape[0]} events remain.')


    return filter