#!/usr/bin/env python3

import os.path
from dataset import *
import util
import numpy as np
import sys
import os.path
import json, pickle
import argparse
from glob import glob
from collections import Mapping, defaultdict
import time

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

def log(logfile, string):
    """ Prints a string and puts into the logfile if present. """
    print(string)
    if logfile is not None:
        logfile.write(str(string) + '\n')


def evaluate_model(model, data_loader, loss_function, logfile=None):
    """ Evaluates the model performance on a dataset (validation or test).
    
    Parameters:
    -----------
    model : torch.nn.Module
        The classifier to evaluate.
    data_loader : torch.utils.data.DataLoader
        Loader for the dataset to evaluate on.
    loss_function : torch.nn.Loss
        The loss function that is optimized.
    logfile : file-like or None
        The file to put logs into.

    Returns:
    --------
    metrics : defaultdict(float)
        The statistics (metrics) for the model on the given dataset.
    """
    model.eval()
    metrics = defaultdict(float)
    y_pred = np.zeros(len(data_loader.dataset))
    y_true = np.zeros(len(data_loader.dataset))
    for batch_idx, (X, D, masks, y_i) in enumerate(data_loader):
        print(f'\rEvaluating {batch_idx + 1} / {len(data_loader)}', end='\r')
        y_pred_i = model(X, D, masks)
        loss = loss_function(y_pred_i, y_i)
        y_pred[batch_idx * data_loader.batch_size : (batch_idx + 1) * data_loader.batch_size] = y_pred_i.data.cpu().numpy().squeeze()
        y_true[batch_idx * data_loader.batch_size : (batch_idx + 1) * data_loader.batch_size] = y_i.data.cpu().numpy().squeeze()
        metrics['loss'] += loss.item()



    metrics['loss'] /= len(data_loader)
    metrics['accuracy'] = accuracy_score(y_true, y_pred >= .5)
    metrics['f1'] = f1_score(y_true, y_pred >= .5)
    metrics['auc'] = roc_auc_score(y_true, y_pred)
    metrics['ppr'] = (y_pred >= .5).sum() / y_pred.shape[0]
    
    values = ' -- '.join(map(lambda metric: f'{metric} : {(metrics[metric]):.4f}', metrics))
    log(logfile, f'\nMetrics: {values}')
    return metrics


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
    model_idx = np.random.randint(10000000000)
    log(logfile, f'### Generating a model id: {model_idx}')
    training_dir = settings['training']['directory'].format(model_idx)
    log(logfile, f'### Saving to {training_dir}')
    os.makedirs(training_dir, exist_ok=True)

    # Create a seed if non given
    if settings['dataset']['seed'] is None:
        settings['dataset']['seed'] = model_idx
        print(f'Seeded with the model id ({model_idx})')

    # Save a copy of the settings
    with open(os.path.join(training_dir, 'config.json'), 'w+') as f:
        json.dump(settings, f)
    
    # Load data
    batch_size = settings['training']['batch_size']
    data = util.dataset_from_config(settings)
    data_train = TorchHD5Dataset(data, 'train')
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=False, collate_fn=data_train.collate)
    data_val = TorchHD5Dataset(data, 'val')
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False, collate_fn=data_val.collate)
    data_test = TorchHD5Dataset(data, 'test')
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, collate_fn=data_test.collate)

    model = util.model_from_config(settings)
    if torch.cuda.is_available():
        model = model.cuda()
        log(logfile, "Training on GPU")
        log(logfile, "GPU type:\n{}".format(torch.cuda.get_device_name(0)))
    if settings['training']['loss'].lower() == 'binary_crossentropy':
        loss_function = nn.functional.binary_cross_entropy
    else:
        raise RuntimeError(f'Unkown loss {settings["training"]["loss"]}')

    optimizer = torch.optim.Adamax(model.parameters(), lr=settings['training']['learning_rate'])
    lr_scheduler_type = settings['training']['learning_rate_scheduler']
    if lr_scheduler_type.lower() == 'reduce_on_plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=settings['training']['learning_rate_scheduler_patience'])
    elif lr_scheduler_type.lower() == 'exponential_decay':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, settings['training']['learning_rate_decay'])
    else:
        raise RuntimeError(f'Unkown learning rate scheduler strategy {lr_scheduler_type}')

    metrics = defaultdict(list)

    epochs = settings['training']['epochs']
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1} / {epochs}, learning rate: {optimizer.param_groups[0]["lr"]}')
        epoch_loss = 0
        epoch_accuracy = 0
        model.train()
        t0 = time.time()
        for batch_idx, (X, D, masks, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(X, D, masks)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            y = y.data.cpu().numpy()
            y_pred = y_pred.data.cpu().numpy()
            y_pred = (y_pred >= .5).astype(np.int)
            epoch_accuracy += (y == y_pred).sum() / y.shape[0]
            dt = time.time() - t0
            eta = dt * (len(train_loader) / (batch_idx + 1) - 1)

            print(f'\r{batch_idx + 1} / {len(train_loader)}: batch_loss {loss.item():.4f} -- epoch_loss {epoch_loss / (batch_idx + 1):.4f} -- epoch acc {epoch_accuracy / (batch_idx + 1):.4f} -- mean of preds {y_pred.mean():.4f} # ETA: {int(eta):6}s      ', end='\r')

        # Validation
        log(logfile, '\n### Validation:')    
        for metric, value in evaluate_model(model, val_loader, loss_function, logfile=logfile).items():
            metrics[metric].append(value)
        # Update learning rate, scheduler uses last accuracy as cirterion
        lr_scheduler.step(metrics['accuracy'][-1])

        # Save model parameters
        checkpoint_path = os.path.join(training_dir, f'model_{epoch + 1}')
        torch.save(model.state_dict(), checkpoint_path)
        log(logfile, f'Saved model to {checkpoint_path}')
    
    log(logfile, '\n### Testing:')
    metrics_testing = evaluate_model(model, test_loader, loss_function, logfile=logfile)

    with open(os.path.join(training_dir, 'stats.pkl'), 'wb') as f:
        pickle.dump({'val' : metrics, 'test' : metrics_testing}, f)

    if logfile is not None:
        logfile.close()





