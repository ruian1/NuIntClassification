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
    dataset_config = config['dataset']
    dataset_type = dataset_config['type'].lower()
    if dataset_type == 'pickle':
        data = PickleDataset(
            path = dataset_config['path'],
            validation_portion = dataset_config['validation_portion'], 
            test_portion = dataset_config['test_portion'],
            shuffle = dataset_config['shuffle']
        )
    elif dataset_type in ('hdf5', 'hd5'):
        data = HD5Dataset(
            dataset_config['path'],
            validation_portion = dataset_config['validation_portion'], 
            test_portion = dataset_config['test_portion'],
            shuffle = dataset_config['shuffle'],
            features = dataset_config['features'],
            graph_features = dataset_config['graph_features'],
            distances = dataset_config['distances'],
            balance_dataset = dataset_config['balance_classes'],
            min_track_length= dataset_config['min_track_length'],
            max_cascade_energy= dataset_config['max_cascade_energy'],
        )
    else:
        raise RuntimeError(f'Unknown dataset type {dataset_type}')
    return data

def model_from_config(config):
    """ Creates a model from a configuration.
    
    Parameters:
    -----------
    config : dict
        The configuration for the model.
    
    Returns:
    --------
    model : keras.models.Model
        A keras model.
    """
    number_input_features = len(config['dataset']['features'])
    model_config = config['model']
    model_type = model_config['type'].lower()
    if model_type == 'gcnn':
        model = GraphConvolutionalNetwork(
            number_input_features,
            units_graph_convolutions = model_config['hidden_units_graph_convolutions'],
            units_fully_connected = model_config['hidden_units_fully_connected'],
            use_batchnorm = model_config['use_batchnorm'],
            dropout_rate = model_config['dropout_rate'],
        )
        num_classes = (model_config['hidden_units_graph_convolutions'] + model_config['hidden_units_fully_connected'])[-1]
    else:
        raise RuntimeError(f'Unkown model type {model_type}')
    return model