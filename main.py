#!/usr/bin/env python3

import tensorflow as tf
import tensorflow.keras as keras
import os.path
from model import GCNN
from dataset import *
import numpy as np
import sys
import os.path
import json
import argparse
from collections import Mapping
tf.enable_eager_execution()

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


class LossLoggingCalback(tf.keras.callbacks.Callback):
    """ Callback for logging the losses at the end of each epoch. """

    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        print(model.evaluate_generator(
            data.get_batches(batch_size=batch_size, dataset='val'),
            steps=data.size(dataset='val') // batch_size))
        baseline_accuracy = data.get_baseline_accuracy(dataset='val')
        print(f'Baseline accuracy {baseline_accuracy}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('settings', help='Settings for the model. See default_settings.json for the default settings. Values are updated with the settings passed here.')
    args = parser.parse_args()

    default_settings_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'default_settings.json')

    with open(default_settings_path) as f:
        settings = json.load(f)
    with open(args.settings) as f:
        dict_update(settings, json.load(f))
    
    # Retrieve the dataset
    dataset_type = settings['dataset']['type'].lower()
    if dataset_type == 'pickle':
        dataset_class = PickleDataset
    elif dataset_type == 'hdf5':
        dataset_class = HD5Dataset
    else:
        raise RuntimeError(f'Unknown dataset type {dataset_type}')
    data = dataset_class(
        path = settings['dataset']['path'],
        validation_portion = settings['dataset']['validation_portion'], 
        test_portion = settings['dataset']['test_portion'],
        shuffle = settings['dataset']['shuffle']
    )

    # Build the model
    model_type = settings['model']['type'].lower()
    if model_type == 'gcnn':
        model = GCNN(
            data.get_number_features(),
            units_graph_convolutions = settings['model']['hidden_units_graph_convolutions'],
            units_fully_connected = settings['model']['hidden_units_fully_connected'],
            use_batchnorm = settings['model']['use_batchnorm'],
            dropout_rate = settings['model']['dropout_rate']
        )
        num_classes = (settings['model']['hidden_units_graph_convolutions'] + settings['model']['hidden_units_fully_connected'])[-1]
    else:
        raise RuntimeError(f'Unkown model type {model_type}')
    
    # Train the model
    checkpoint_path = os.path.join(settings['training']['checkpoint_directory'], 'cp-{epoch:04d}.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        period = settings['training']['checkpoint_period']
    )
    optimizer = tf.keras.optimizers.Adam()
    if num_classes == 1:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    model.compile(optimizer=optimizer, 
                loss=loss,
                metrics=settings['training']['metrics'])
    if settings['training']['use_class_prior']:
        class_prior = data.get_class_prior()
    else:
        class_prior = None
    batch_size = settings['training']['batch_size']
    model.fit_generator(
        data.get_batches(batch_size=batch_size, dataset='train'), 
        steps_per_epoch = int(np.ceil(data.size(dataset='train') / batch_size)),
        epochs = settings['training']['epochs'],
        callbacks = [checkpoint_callback, LossLoggingCalback()],
        class_weight = class_prior
        )
    
    # Evaluate on test set
    print('### Test dataset results:')
    print(model.evaluate_generator(
            data.get_batches(batch_size=batch_size, dataset='test'),
            steps=data.size(dataset='test') // batch_size))
    baseline_accuracy = data.get_baseline_accuracy(dataset='test')
    print(f'Baseline accuracy {baseline_accuracy}')
    

