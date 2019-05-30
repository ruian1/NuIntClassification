import json
from dataset import *
from model import *
from collections import Mapping

def dict_update(d, u):
    """ Recursively updates a dictionary with another. Used for parsing settings for training.
    
    Parameters:
    -----------
    d : dict
        The dict to update.
    u : dict
        The dict that contains keys that should be updated in d.
    """
    for key in u:
        if key in d:
            if isinstance(d[key], Mapping):
                dict_update(d[key], u[key])
            else:
                d[key] = u[key]
        else:
            raise RuntimeError(f'Unkown setting {key}')

def dataset_from_config(config):
    """ Creates a dataset from a configuration file. 
    
    Parameters:
    -----------
    config : dict
        The configuration dict.
    
    Returns:
    --------
    data : dataset.Dataset
        The dataset to train / test on.
    """
    dataset_type = config['type'].lower()
    if dataset_type == 'pickle':
        data = PickleDataset(
            path = config['path'],
            validation_portion = config['validation_portion'], 
            test_portion = config['test_portion'],
            shuffle = config['shuffle']
        )
    elif dataset_type in ('hdf5', 'hd5'):
        data = HD5Dataset(
            config['path'],
            validation_portion = config['validation_portion'], 
            test_portion = config['test_portion'],
            shuffle = config['shuffle'],
            features = config['features'],
        )
    else:
        raise RuntimeError(f'Unknown dataset type {dataset_type}')
    return data

def model_from_config(config, number_input_features):
    """ Creates a model from a configuration.
    
    Parameters:
    -----------
    config : dict
        The configuration for the model.
    num_input_features : int
        The number of input features.
    
    Returns:
    --------
    model : keras.models.Model
        A keras model.
    """
    model_type = config['type'].lower()
    if model_type == 'gcnn':
        model = GCNN(
            number_input_features,
            units_graph_convolutions = config['hidden_units_graph_convolutions'],
            units_fully_connected = config['hidden_units_fully_connected'],
            use_batchnorm = config['use_batchnorm'],
            dropout_rate = config['dropout_rate']
        )
        num_classes = (config['hidden_units_graph_convolutions'] + config['hidden_units_fully_connected'])[-1]
    else:
        raise RuntimeError(f'Unkown model type {model_type}')
    return model