#!/usr/bin/env python3

import tensorflow as tf
import tensorflow.keras as keras
import os.path
from dataset import *
import util
import numpy as np
import sys
import os.path
import json
import argparse
from collections import Mapping
tf.enable_eager_execution()

class LossLoggingCalback(tf.keras.callbacks.Callback):
    """ Callback for logging the losses at the end of each epoch. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_history = []

    def on_epoch_end(self, epoch, logs=None):
        print(model.evaluate_generator(
            data.get_batches(batch_size=batch_size, dataset='val'),
            steps=int(np.ceil(data.size(dataset='val') // batch_size))))
        y_pred = model.predict_generator(data.get_batches(batch_size=batch_size, dataset='val'),
                       steps=int(np.ceil(data.size(dataset='val') / batch_size)))
        print(np.unique(y_pred), np.unique(y_pred).shape)

        baseline_accuracy = data.get_baseline_accuracy(dataset='val')
        
        print(f'Baseline accuracy {baseline_accuracy}')

    def on_train_batch_end(self, batch, logs=None):
        if logs is not None:
            self.training_history.append([
                logs[metric] for metric in logs
            ])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('settings', help='Settings for the model. See default_settings.json for the default settings. Values are updated with the settings passed here.')
    args = parser.parse_args()

    default_settings_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'default_settings.json')

    with open(default_settings_path) as f:
        settings = json.load(f)
    with open(args.settings) as f:
        util.dict_update(settings, json.load(f))

    # Set up the directory for training and saving the model
    model_idx = np.random.randint(1000000)
    print(f'### Generating a model id: {model_idx}')
    training_dir = settings['training']['directory'].format(model_idx)
    print(f'### Saving to {training_dir}')
    os.makedirs(training_dir, exist_ok=True)

    # Save a copy of the settings
    with open(os.path.join(training_dir, 'config.json'), 'w+') as f:
        json.dump(settings, f)
    
    checkpoint_path = os.path.join(training_dir, 'checkpoint', 'cp-{epoch:04d}.ckpt')
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_weights_only=True,
            period = settings['training']['checkpoint_period']
        )
    logging_callback = LossLoggingCalback()

    data = util.dataset_from_config(settings)
    model = util.model_from_config(settings)    
    
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
    
    # Evaluate on test set
    print('### Test dataset results:')
    print(model.evaluate_generator(
            data.get_batches(batch_size=batch_size, dataset='test'),
            steps=data.size(dataset='test') // batch_size))
    baseline_accuracy = data.get_baseline_accuracy(dataset='test')
    print(f'Baseline accuracy {baseline_accuracy}')

    # Save the model and history
    model_path = os.path.join(training_dir, 'model_weights.h5')
    model.save_weights(model_path)
    print(f'### Saved model weights to {model_path}')

    # Save the history
    history = np.array(logging_callback.training_history)
    history_path = os.path.join(training_dir, 'history.npy')
    np.save(history_path, logging_callback.training_history)
    print(f'### Saved history to {history_path}')

    

