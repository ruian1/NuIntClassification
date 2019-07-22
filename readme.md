# Neutrino Interaction Classification
Implementation of a Graph Convolutional Network (GCN) to identify low-energy track-like events in IceCube. The implementation follows [https://arxiv.org/abs/1809.06166].

## Creating data

Data is retrieved directly from i3 files (Monte Carlo Simulation or real-world). Since models are trained on hd5 files, the data needs to be preprocessed (also attributes graphs have to be extracted from the raw pulse event data). Scripts in `create_dataset/` implement this process.

1. `create_dataset.py` extracts graphs from an i3 file and creates a corresponding hd5 file
2. `concatenate_datasets.py` discards irrelevant meta data of hd5 files and combines multiple hd5 files into one huge dataset.
3. `transform_datasets.py` shuffles and splits a large dataset into training, validation and testing data

## Training a model

Model architectures, as well as the data to be used and hyperparameters for training (such as the learning rate) are all specified in a JSON file. Examples can be found in `settings/`. Currently, models that only process vertex features as well as models that also take graph features into account are possible. Defaults for each possible setting exist and can be found in `default_settings.json`.

To train a model, the command line interface `train.py` is provided. It creates a directory for the experiment, where model parameters, log files and the overall detailed training metrics for each epoch can be found after training. Also, a copy of the model architecture (i.e. the JSON file) is placed there.

## Evaluating a model

To evaluate a model, currently the command line interface `evaluate.py` is provided. Using a pretrained model (i.e. its JSON file and model parameters) it can evaluate the performance on any other hd5 dataset file. The predictions of each event are stored in a pickle file, that contains a two-level dictionary mapping filenames and event ids of each event to the model's prediction score.

## Documentation

A detailed explanation of the code and the overall idea, as well as future experiments can be found in `doc.md`

