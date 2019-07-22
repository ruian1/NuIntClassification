#!/usr/bin/env python3

import argparse
import sys
import json
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

import util
from dataset import *


def load_model(config, model_parameters):
    """ Loads an already trained pytorch model. 
    
    Parameters:
    -----------
    config : dict
        Configuration for the model.
    model_parameters : str
        Path to the model parameters to be loaded. 

    Returns:
    --------
    model : torch.module
        The model.
    """
    model = util.model_from_config(config)
    if torch.cuda.is_available():
            model = model.cuda()
    model.load_state_dict(torch.load(model_parameters))
    return model


def load_dataset(config, dataset_path):
    """ Creates a hd5 dataset that matches the configuration used to train the model.
    
    Parameters:
    -----------
    config : dict
        Configuration for the model.
    dataset_path : str
        Path to the hd5 dataset.
    
    Returns:
    --------
    dataset : dataset.ShuffledTorchDataset
        The dataset to evaluate the model on.
    """
    dataset_config = config['dataset']
    dataset_type = dataset_config['type'].lower()
    if dataset_type in ('hdf5', 'hd5'):
        dataset = ShuffledGraphTorchHD5Dataset(
            dataset_path,
            features = dataset_config['features'],
            coordinates = dataset_config['coordinates'],
            balance_dataset = False,
            min_track_length = None,
            max_cascade_energy = None,
            flavors = None,
            currents = None,
            class_weights = 'balanced',
            close_file = False,
        )
    elif dataset_type in ('hdf5_graph_features', 'hd5_graph_features'):
        dataset = ShuffledGraphTorchHD5DatasetWithGraphFeatures(
            dataset_path,
            features = dataset_config['features'],
            coordinates = dataset_config['coordinates'],
            graph_features = dataset_config['graph_features'],
            balance_dataset = False,
            min_track_length = None,
            max_cascade_energy = None,
            flavors = None,
            currents = None,
            class_weights = 'balanced',
            close_file = False,
        )
    else:
        raise RuntimeError(f'Unkown dataset type {dataset_type}')
    return dataset
        

def evaluate(config, model, dataset, verbose=True):
    """ Evaluates the model on an hd5 dataset. 
    
    Parameters:
    -----------
    config : dict
        Configuration for the model.
    model : torch.nn.module
        The model that is used for evaluation (with parameters already loaded).
    dataset : dataset.ShuffledTorchDataset
        The dataset the model is supposed to be evaluated on.

    Returns:
    --------
    scores : dict
        A dict that maps a tuple filepaths to dicts, which map EventIDs to predicted scores.
    """
    # Create a data loader
    batch_size = config['training']['batch_size']
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate, drop_last=False)

    scores = defaultdict(dict)
    model.eval()
    
    accuracies = []

    for batch_idx, (inputs, y_i, _) in enumerate(data_loader):
        if verbose: print(f'\rEvaluating {batch_idx + 1} / {len(data_loader)}', end='\r')
        i_start, i_end = batch_idx * data_loader.batch_size, (batch_idx + 1) * data_loader.batch_size
        y_pred_i = model(*inputs).data.cpu().numpy().squeeze()
        y_i = y_i.data.cpu().numpy().squeeze()
        accuracies.append(accuracy_score(y_i, y_pred_i >= .5))
        
        # Get the filenames and event ids for the batch
        filenames = dataset.file['filename'][i_start : i_end]
        event_idxs = dataset.file['EventID'][i_start : i_end]
        for score, filename, event_idx in zip(y_pred_i, filenames, event_idxs):
            scores[filename][event_idx] = score
        
    print(f'Overall accuracy (if available): {np.array(accuracies).mean()}')
    return dict(scores)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Path to the hd5 dataset to evaluate data on.')
    parser.add_argument('model', help='Parser to the model\'s configuration file.')
    parser.add_argument('model_parameters', help='Parser to the model parameters.')
    parser.add_argument('output', help='Output pickle file.')

    args = parser.parse_args()

    # Load the configuration
    with open('default_settings.json') as f:
        config = json.load(f)
    with open(args.model) as f:
        util.dict_update(config, json.load(f))
    
    model = load_model(config, args.model_parameters)
    dataset = load_dataset(config, args.dataset)
    scores = evaluate(config, model, dataset, verbose=True)
    
    with open(args.output, 'wb') as f:
        pickle.dump(scores, f)

    
    


