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
import torch.utils.tensorboard as tb
import datetime

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from scipy.stats import spearmanr

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn



def get_metrics(y_true, y_pred):
    """ Calculates all desired metrics for a given set of predictions and ground truths.
    
    Parameters:
    -----------
    y_true : ndarray, shape [N]
        Ground truth class labels.
    y_pred : ndarray, shape [N]
        Predicted class probabilities or hard labels.
        
    Returns:
    --------
    metrics : defaultdict
        A dict containing values for all metrics. 
    """
    metrics = defaultdict(float)
    metrics['accuracy'] = accuracy_score(y_true, y_pred >= .5)
    try: 
        metrics['auc'] = roc_auc_score(y_true, y_pred)
    except: 
        # AUC is not defined, if only one class is present 
        metrics['auc'] = np.nan
    metrics['ppr'] = (y_pred >= .5).sum() / y_pred.shape[0]
    return metrics
    

def evaluate_model(model, data_loader, loss_function, prefix, iteration, tb_writer=None):
    """ Evaluates the model performance on a dataset (validation or test).
    
    Parameters:
    -----------
    model : torch.nn.Module
        The classifier to evaluate.
    data_loader : torch.utils.data.DataLoader
        Loader for the dataset to evaluate on.
    loss_function : torch.nn.Loss
        The loss function that is optimized.
    prefix : str
        The name of the evaluation, i.e. 'validation' or 'test'
    iteration : int
        The current evaluation iteration.
    tb_writer : SummaryWriter or None
        Tensorboard writer for logging.

    Returns:
    --------
    metrics : defaultdict(float)
        The statistics (metrics) for the model on the given dataset.
    """
    model.eval()
    metrics = defaultdict(float)
    y_pred = np.zeros(len(data_loader.dataset))
    y_true = np.zeros(len(data_loader.dataset))
    total_loss = 0
    for batch_idx, (inputs, y_i, weights) in enumerate(data_loader):
        print(f'\rEvaluating {batch_idx + 1} / {len(data_loader)}', end='\r')
        y_pred_i = model(*inputs)
        loss = loss_function(y_pred_i, y_i, weights)
        y_pred[batch_idx * data_loader.batch_size : (batch_idx + 1) * data_loader.batch_size] = y_pred_i.data.cpu().numpy().squeeze()
        y_true[batch_idx * data_loader.batch_size : (batch_idx + 1) * data_loader.batch_size] = y_i.data.cpu().numpy().squeeze()
        total_loss += loss.item()

    metrics = get_metrics(y_true, y_pred)
    metrics['loss'] = total_loss / len(data_loader)

    if tb_writer:
        tb_writer.add_scalar(f'{prefix} loss', metrics['loss'], iteration)
        tb_writer.add_scalar(f'{prefix} accuracy', metrics['accuracy'], iteration)
        tb_writer.add_scalar(f'{prefix} mean trackness', y_pred.mean(), iteration)
        energy = np.array(data_loader.dataset.file['NeutrinoEnergy'])
        tb_writer.add_scalar(f'{prefix} spearman correlation energy - trackness', spearmanr(energy, y_pred.flatten())[0], iteration)
    
    values = ' -- '.join(map(lambda metric: f'{metric} : {(metrics[metric]):.4f}', metrics))
    print(f'\nMetrics: {values}')
    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Configuration for the training. See default_settings.json for the default settings. Values are updated with the settings passed here.')
    parser.add_argument('--array', help='If set, the "config" parameter refers to a regex. Needs the file index parameter.', action='store_true')
    parser.add_argument('-i', type=int, help='Index of the file in the directory to use as configuration file. Only considered if "--array" is set.')
    parser.add_argument('--verbose', help='If set, logs to stdout.', action='store_true')
    args = parser.parse_args()


    verbose = args.verbose
    default_settings_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'default_settings.json')
    if args.array:
        config_path = glob(args.config)[args.i]
    else:
        config_path = args.config
    
    with open(default_settings_path) as f:
        settings = json.load(f)
    with open(config_path) as f:
        util.dict_update(settings, json.load(f))



    print(f'### Training according to configuration {config_path}')

    # Set up the directory for training and saving the model
    model_idx = np.random.randint(10000000000)
    print(f'### Generating a model id: {model_idx}')
    training_dir = settings['training']['directory'].format(model_idx)
    print(f'### Saving to {training_dir}')
    os.makedirs(training_dir, exist_ok=True)

    # Create a seed if non given
    if settings['seed'] is None:
        settings['seed'] = model_idx
        print(f'Seeded with the model id ({model_idx})')

    np.random.seed(settings['seed'] & 0xFFFFFFFF)
    torch.manual_seed(settings['seed'] & 0xFFFFFFFF)

    # Tensorboard logging
    if settings['training']['logdir']:
        date = datetime.datetime.now()
        timestamp = date.strftime(f"%d-%b-%Y_%H.%M.%S")
        logdir = os.path.join(settings['training']['logdir'], f'{timestamp}_{model_idx}')
        os.makedirs(logdir)
        tb_writer = tb.SummaryWriter(logdir)
    else:
        tb_writer = None

    #tb_writer.add_hparams(hparam_dict={'configuration' : str(settings)}, metric_dict={})
    tb_writer.add_text('configuration', str(settings))

    # Save a copy of the settings
    with open(os.path.join(training_dir, 'config.json'), 'w+') as f:
        json.dump(settings, f)
    
    # Load data
    batch_size = settings['training']['batch_size']

    data_train, data_val, data_test = util.dataset_from_config(settings)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=settings['dataset']['shuffle'], collate_fn=data_train.collate, drop_last=False)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=settings['dataset']['shuffle'], collate_fn=data_val.collate, drop_last=False)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=settings['dataset']['shuffle'], collate_fn=data_test.collate, drop_last=False)

    model = util.model_from_config(settings)
    if torch.cuda.is_available():
        model = model.cuda()
        print("Training on GPU")
        print("GPU type:\n{}".format(torch.cuda.get_device_name(0)))
    if settings['training']['loss'].lower() == 'binary_crossentropy':
        loss_function = nn.functional.binary_cross_entropy
    else:
        raise RuntimeError(f'Unkown loss {settings["training"]["loss"]}')

    optimizer = torch.optim.Adamax(model.parameters(), lr=settings['training']['learning_rate'])
    lr_scheduler_type = settings['training']['learning_rate_scheduler']
    if lr_scheduler_type.lower() == 'reduce_on_plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
        patience=settings['training']['learning_rate_scheduler_patience'], min_lr=settings['training']['min_learning_rate'])
    elif lr_scheduler_type.lower() == 'exponential_decay':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, settings['training']['learning_rate_decay'])
    elif lr_scheduler_type.lower() == 'constant':
        lr_scheduler = None
    else:
        raise RuntimeError(f'Unkown learning rate scheduler strategy {lr_scheduler_type}')

    validation_metrics = defaultdict(list)
    training_metrics = defaultdict(list)

    print(f'Training on {len(data_train)} samples.')

    epochs = settings['training']['epochs']
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1} / {epochs}, learning rate: {optimizer.param_groups[0]["lr"]}')
        running_loss = 0
        running_accuracy = 0
        model.train()
        t0 = time.time()
        for batch_idx, (inputs, targets, weights) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(*inputs)
            loss = loss_function(y_pred, targets, weight=weights)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            targets = targets.data.cpu().numpy()
            y_pred = y_pred.data.cpu().numpy()
            batch_metrics = get_metrics(targets, y_pred)
            for metric, value in batch_metrics.items():
                training_metrics[metric].append(value)
            running_accuracy += batch_metrics['accuracy']
            # Estimate ETA
            dt = time.time() - t0
            eta = dt * (len(train_loader) / (batch_idx + 1) - 1)
            if verbose:
                print(f'\r{batch_idx + 1} / {len(train_loader)}: batch_loss {loss.item():.4f} -- epoch_loss {running_loss / (batch_idx + 1):.4f} -- epoch acc {running_accuracy / (batch_idx + 1):.4f} -- mean of preds / targets {y_pred.mean():.4f} / {targets.mean():.4f} # ETA: {int(eta):6}s      ', end='\r')
            if tb_writer:
                train_iteration = epoch * len(train_loader) + batch_idx
                tb_writer.add_scalar('train loss', loss.item(), train_iteration)
                tb_writer.add_scalar('train accuracy', batch_metrics['accuracy'], train_iteration)
                tb_writer.add_scalar('mean trackness', y_pred.mean(), train_iteration)
                energy = np.array(data_train.file['NeutrinoEnergy'][data_train._idxs[batch_idx * batch_size : (batch_idx + 1) * batch_size]])
                tb_writer.add_scalar('spearman correlation energy - prediction', spearmanr(energy, y_pred.flatten())[0], train_iteration)

        # Validation
        print('\n### Validation:')    
        for metric, value in evaluate_model(model, val_loader, loss_function, 'validation', epoch, tb_writer=tb_writer).items():
            validation_metrics[metric].append(value)
        # Update learning rate, scheduler uses last accuracy as cirterion
        if lr_scheduler:
            lr_scheduler.step(validation_metrics['accuracy'][-1])

        # Save model parameters
        checkpoint_path = os.path.join(training_dir, f'model_{epoch + 1}')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Saved model to {checkpoint_path}')
    
    print('\n### Testing:')
    testing_metrics = evaluate_model(model, test_loader, loss_function, 'test', 0, tb_writer=tb_writer)

    with open(os.path.join(training_dir, 'stats.pkl'), 'wb') as f:
        pickle.dump({'train': training_metrics, 'val' : validation_metrics, 'test' : testing_metrics}, f)






