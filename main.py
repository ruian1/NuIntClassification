#!/usr/bin/env python3

import tensorflow as tf
import tensorflow.keras as keras
import os.path
from dataset import *
import util
import numpy as np
import sys
import os.path
import json, pickle
import argparse
from glob import glob
from collections import Mapping
tf.enable_eager_execution()

def log(logfile, string):
    """ Prints a string and puts into the logfile if present. """
    print(string)
    if logfile is not None:
        logfile.write(str(string) + '\n')

class LossLoggingCalback(tf.keras.callbacks.Callback):
    """ Callback for logging the losses at the end of each epoch. """

    def __init__(self, model, data, logfile, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_history = []
        self.validation_history = []
        self.model = model
        self.data = data
        self.logfile = logfile
        self.batch_size = batch_size

    def log(self, string):
        log(self.logfile, string)

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.model.evaluate_generator(
            self.data.get_batches(batch_size=self.batch_size, dataset='val'),
            steps=int(np.ceil(self.data.size(dataset='val') // self.batch_size))
        )
        self.validation_history.append(metrics)
        self.log(f'Metrics: {metrics}')
        predictions = self.model.predict_generator(
            self.data.get_batches(batch_size=self.batch_size, dataset='val'),
            steps=int(np.ceil(self.data.size(dataset='val') / self.batch_size))
        )
        unique_predictions = np.unique(predictions).shape[0] / predictions.shape[0]
        self.log(f'Predictions: {predictions}, ({unique_predictions}% of {predictions.shape[0]} unique predictions)')
        positives = (predictions > 0.5).sum() / predictions.shape[0]
        self.log(f'Positive Rate: {positives}')
        baseline_accuracy = self.data.get_baseline_accuracy(dataset='val')
        self.log(f'Baseline accuracy {baseline_accuracy}')

    def on_train_batch_end(self, batch, logs=None):
        if logs is not None:
            self.training_history.append([
                logs[metric] for metric in logs
            ])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Configuration for the training. See default_settings.json for the default settings. Values are updated with the settings passed here.')
    parser.add_argument('--array', help='If set, the "config" parameter refers to a regex. Needs the file index parameter.', action='store_true')
    parser.add_argument('-i', type=int, help='Index of the file in the directory to use as configuration file. Only considered if "--array" is set.')
    args = parser.parse_args()

    default_settings_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'default_settings.json')
    if args.array:
        config_path = glob(args.config)[args.i]
    else:
        config_path = args.config
    
    with open(default_settings_path) as f:
        settings = json.load(f)
    with open(config_path) as f:
        util.dict_update(settings, json.load(f))

    # Create a logfile
    if settings['training']['logfile']:
        logfile = open(settings['training']['logfile'], 'w+')
    else:
        logfile = None

    log(logfile, f'### Training according to configuration {config_path}')

    # Set up the directory for training and saving the model
    model_idx = np.random.randint(1000000)
    log(logfile, f'### Generating a model id: {model_idx}')
    training_dir = settings['training']['directory'].format(model_idx)
    log(logfile, f'### Saving to {training_dir}')
    os.makedirs(training_dir, exist_ok=True)

    # Save a copy of the settings
    with open(os.path.join(training_dir, 'config.json'), 'w+') as f:
        json.dump(settings, f)
    
    checkpoint_path = os.path.join(training_dir, 'checkpoint', 'cp-{epoch:04d}.ckpt')
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_weights_only=True,
            period = settings['training']['checkpoint_period']
        )

    data = util.dataset_from_config(settings)
    model = util.model_from_config(settings)

    logging_callback = LossLoggingCalback(model, data, logfile, settings['training']['batch_size'])
    
    optimizer = tf.keras.optimizers.Adam()
    loss = settings['training']['loss']
    model.compile(optimizer=optimizer, 
                loss=loss,
                metrics=settings['training']['metrics'])
    if settings['training']['use_class_weights']:
        class_weights = data.get_class_weights()
    else:
        class_weights = None
    batch_size = settings['training']['batch_size']
    model.fit_generator(
        data.get_batches(batch_size=batch_size, dataset='train'), 
        steps_per_epoch = int(np.ceil(data.size(dataset='train') / batch_size)),
        epochs = settings['training']['epochs'],
        callbacks = [checkpoint_callback, logging_callback],
        class_weight = class_weights
        )
    
    # Save the model
    model_path = os.path.join(training_dir, 'model_weights.h5')
    model.save_weights(model_path)
    log(logfile, f'### Saved model weights to {model_path}')

    # Evaluate on test set
    log(logfile, '### Test dataset results:')
    test_metrics = model.evaluate_generator(
            data.get_batches(batch_size=batch_size, dataset='test'),
            steps=data.size(dataset='test') // batch_size)
    log(logfile, test_metrics)
    baseline_accuracy = data.get_baseline_accuracy(dataset='test')
    log(logfile, f'Baseline accuracy {baseline_accuracy}')

    # Save the history
    history = {
        'training' : logging_callback.training_history,
        'validation' : logging_callback.validation_history,
        'test' : test_metrics,
    }
    history_path = os.path.join(training_dir, 'history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history)
    log(logfile, f'### Saved training history to {history_path}')

    if logfile is not None:
        logfile.close()

    

