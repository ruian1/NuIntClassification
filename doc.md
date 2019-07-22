# Neutrino Interaction Classification using GCNs

When Neutrinos interact with matter in IceCube, secondary particles can be created, that create light, which is then seen by the optical modules (DOMs) inside the detector.

Depending on the flavor of the Neutrino and the interaction type (charged or neutral current), different processes can occur. For Muon Neutrinos that undergo a charged current, muons that live long enough to trace out a recognizable track, are produced. That is, apart from the light emitted by the interaction and the resulting shower, the muon emits Cherenkov light along its path. This "signature" is refered to as track-like. All other flavors and even Muon Neutrinos that undergo a neutral current interaction, only produce a cascade of light (which appears as a somewhat isotropic light sphere to the detector). This "signature" is refered to as cascade-like. The goal of this project is to distinguish between those two event signatures using the detectors data for low energy neutrinos, where the tracks are especially short and hard to distinguish from cascade-only topologies.

# IceCube Detector Layout and Graphs

In order to apply Machine Learning (i.e. Neural Nets in this case) to this problem, a challange is how to represent the detector data. i3-files record all pulses of each optical module (DOM) which saw light during an event. Each event corresponds to one neutrino interaction. Inside the i3 file each DOM that was "active" lists all pulses that occured, giving information about:
- the charge (which corresponds to the amount of light seen)
- the time of the pulse

Since however the number of DOMs that register any light at all varies from event to event. Also, the detectors shape is highly irregular: A 3d hexagonal grid spreads out the optical modules along strings. These strings are evenly spaced out in general, with the exception of "DeepCore", an area with a denser string topology inside lower part of the detector's volume.

Neural Networks however work on regular and "well-shaped" data, like pictures of fixed size or vector data. Thus, one has to come up with a somewhat regular representation of this kind of data as well as a Neural Network architecture, that supports irrgeularities.

## Graphs

A Graph is a collection of N vertices, which are connected by edges. This forms some kind of "network-structure". A intuitive example would be a social network: Persons would correspond to vertices here,and the interactions between those people (e.g. friendships) would link them via edges. The edges can be assigned weights, which give information about the strongness of a link between vertices. Also each vertex can have a real-valued set of attributes.

The edge weights are stored as an NxN *adjacency matrix*, which simply lists all edge weights between each possible vertex pair. The attributes of the vertices are stored as an NxD *feature matrix*.

## i3 Events as Graphs

Each event will be transformed into a graph following this procedure:
- Each active DOM corresponds to a vertex
- Each DOM is assigned attributes that correspond to the pulse series it recorded:
   - Time of the first, last and maximal (w.r.t. to charge) pulse
   - Charge of the first, last and maximal pulse

Edges are created between each pair of DOMs, i.e. the graph structure is highly dense. The edge weights correspond to the spatial distance between the DOMs, by applying a Gaussian Kernel to the absolute distance between them. The variance of the kernel is a learnable parameter.

# Graph Convolutional Neural Networks

Graph Neural Networks (GCNs) take graphs as input values. As for vanilla Neural Networks, they are a series of layers stacked upon on another, i.e. the output of one layer is fed as input to the next layer.

Each layer implements a graph convolution, which first applies a linear transformation to the vertex feature it received as input. Afterwards, each vertex aggregates the transformed features of all its adjacent neighbouring vertices by summing them up according to the edge weights. Thus, the adjacency matrix (the edge weights) have to be normalized to sum up to 1 for each vertex. After the aggregation, a non-linearity is applied to the vertex features, which are then outputted by the layer.

Therefore, a Graph Convolution receives an attributed graph as input and outputs a graph with the same topology but different attributes for the vertices. The overall GCN outputs a graph as well since it is a series of graph convolutions.

## Dropout and BatchNorm and Residual Blocks

Each Graph Convolution also is able to perform Batch Normalization and Dropout, which have proven useful to other Deep Learning tasks. BatchNorm scales the input features of each vertex along a mini-batch to have mean of zero and a standard deviation of 1 along each feature and applies a learnable affine transformation afterwards. Dropout randomly disables some neurons to prevent overfitting. Residual blocks add the input of a Graph Convolution layer directly to the output of a Graph Convolution layer. The order in which these are applied within a Graph Convolution layer is:
1. Graph Convolution
2. Batch Normalization
3. Non-Linear Activation (ReLU)
4. Residual
5. Dropout

## Classification

Since a series of graph convolutions yields yet another graph, it is not appearent how to arrive at a classification, i.e. the track-probability of the transformed graph. Therefore, after the graph convolutions, an average pooling is performed over all vertices. The resulting embedding in vector is then fed into another vanilla neural network to arrive at a classification. Usually, this network only contains a single layer, i.e. it corresponds to logistic regression.

## Event Features

In addition to the DOM (vertex) based features, the graph itself can be attributed as well, using the reconstruction of a track that is available in the i3 files already. Even non-track events contain this reconstruction (which of course is then fairly off). The reconstruction is processed by a vanilla Neural Network. Its output is appended to the representation of the graph which was received by the average pooling of the output of the GCN. This way, the reconstruction can be incorporated (naively) into the classification procedure. It may also be beneficial to use more then a single layer for the overall classification in order to reasonably combine the information.

## Auxiliary Learning

Another approach (which however is not fully implemented yet) is to let the network predict the energy and direction of the primary neutrino as well as its "trackness". The loss function is a weighted sum of the binary crossentropy loss obtained by the classification as well as a mean-squared-error loss obtained by the regression task of predicting these features. After concatenating the output of the GCN with the output of the NN that processes the reconstruction, two NNs are used to either classify the event or predict the primary particle's properties.

# Dataset

The dataset is obtained following multiple steps, starting at the i3 files, which contain simulations of neutrinos for IceCube. Only for simulated data ground truth class labels are available, so one has to resort to Monte Carlo datasets.

All i3 files are aggregated into a single large hdf5 file that contains all events that are available. The python scripts `create_dataset/create_dataset.py` and `create_dataset/create_recurrent_dataset.py` are responsible for parsing a single i3 file each and storing the ouput into an hdf5 file each. For each i3 file a single hdf5 file is created. Since parsing an i3 file and handling its content requires the `icetray` package, it has to be run within a singularity environment. The slurm script as well as a shell script to wrap this process are designed to operate on computecanada's `cedar` cluster.

Jobs for creating datasets (`create_dataset/create_dataset_slurm_job.sh` and `create_dataset/create_recurrent_dataset_slurm_job.sh`) are supposed to be run as array-jobs, where the number of jobs should match the number of i3 files to be parsed.

The script `create_dataset/concatenate_datasets.py` gets rid of all metadata and creates one single large hdf5 datafile, which contains all events.

In order to speed up the training, the dataset is split into training, validation and testing data beforehand. 10% of the entire data are used for validation, another 10% for testing, such that 80% of all events remain for training. The splitting is performed using `create_dataset/transform_dataset.py`. This script also shuffles the data.

## DOM Coordinates

The coordinates of each DOM are normalized, before they are fed into the GCN. They are centered to the mean of the entire event. This mean can either be a weighted mean (by charges) or just an unweighted one. An empircally obtained value is used to scale the coordinates along each axis, such that they approximately fit into a distribution, that somewhat resembles the standard normal distribution.

## Pulse Times

Another way to incorporate the reconstruction is, to calculate physically when a DOM is expected to recognize charge first, assuming the length and direction of the reconstructed track are correct. It is then possible, to calculate a difference between the time of actually observing any light and this expected time value. The idea is, that for tracks, the difference should not be too large, while for cascade-like events, there is a huge discrepancy to expect. These attributes are also available as features in the hdf5 file, as the scripts for creating datasets calculate these values as well.

A few remarks regarding this calculation: Since for low energy events the tracks are usually short, the entire track is contained inside the detector volume. Thus, not every DOM is expected to see light at all. Therefore, one can also calculate the expected travel time of light from the interaction vertex (where a potentially small cascade occures as well) to any DOM. The smaller arrival time of the one of the track and the cascade is used as a reference value.

Furthermore, light from the track may scatter. Therefore, DOMs, which actually see light originating from a track, might receive a NaN expected time as well. To counteract this issue, the following is done: The track length is temporarily set to "infinite" and the closest point on the track, where the light that a DOM could have seen, is reconstructed. Points which occur after the interaction vertex on the track (in the direction of the track) are kept, since the DOM may have seen light from this point due to scattering. The calcualted expected travel time of course is not truthful there, but at least gives a good approximation. If the estimated point on the track, where the light a DOM could have seen, however is before the interaction vertex (in the direction of the track), the time is discarded and only the cascade time is considered.


# Training

Each training setup is linked to a json configuration file, that specifies the model architecture (number of layers, weather to use batch normalization, dropout, etc.), as well as the data used, together with the learning rate, learning rate scheduler and logging. `default_settings.json` contains default values for all settings. If a setting does not appear in a configuration file, the deafults are read out from this file.

Since the data contains almost two million events, the data loading process is a huge bottleneck of the entire training. To counteract this issue, whenever an hdf5 file is loaded, a numpy memory map (`memmap`) is generated. This memmap enables faster access to the attributes, since it already is shaped as numpy array. Also, over several training setups, these memmaps will be cached. For a certain setting of data, a hash is generated, which can be used to identify the same dataset again in the future. Therefore, if the data setup does not change, loading the data only takes a few seconds. When new data is however used, or settings to the dataset are, there is a very large overhead generated by the creation of those memmaps. They are stored in `memmaps/`.

Each time an experiment is conducted (i.e. a model is trained), the model itself is identified with a randomly generated id. This id can be used to re-load the model again later on and evalute its performance in detail. The model parameters as well as logfiles are stored in `training/hd5_{mode_id}/` for each model.

## Model architectures

Typically models contained 5-8 hidden layers, each with a hidden embedding size of 64. Batchnormalization was used as well as dropout with dropout-rates between 0.3 and 0.5. Residual connections were enabled. Performance did somewhat decrease for models without Batchnormalization, also the learning progress was much slower. Overall however, the number of hidden layers did not yield any significant performance discrepancies.

## Event filtering

The training data can be filtered before a model is actually trained. This may be beneficial for certain scenarios (see below). Currently supported filters are:
- Length of the true track (only applied to NuMu CC events)
- Energy used up in the cascade at the interaction vertex (only applied to NuMu CC events)
- neutrino flavor
- neutrino interaction type
- minimal total energy
- maximal total energy

Since each filter results in a different training dataset, when a filter changes, new memmaps are created. By default, no filter is appleid to the validation and testing data (which however an be adapted).

# Model performance

During training, the performance of a model on the validation data is logged in terms of accuracy and positive prediction rate. The latter is used to verfiy, that the model did not degrade to a "always predict the same class" classifier.

To further investigate a models preformance and predictions, a notebook `baselines.ipynb` is available which compares the model the Likelihood Baseline currently used to identify tracks. It re-evaluates the model on test data and plots the predicted fraction of tracks per neutrino flavor and interaction type. Also, the correlation between an input feature and the predicted trackness is plotted, to observe, if the model has learned any "simple" classification rule.

To investigate the dataset, another notebook `data.ipynb` can be used to plot the transformed DOM coordinates together with the adjacency matrix of any event. This can be used to investigate in details, which kind of events the model did not classify correctly.

# Results

## Models trained on an unbalanced dataset

Models that were trained on a dataset that contained a class imbalance (tracks vs. non-tracks) typically degraded into one-class-predictors, which only assinged "non-track" to each sample. Thus, two methods to counteract this, were implemented:
- Discarding random samples of the class with more samples such that they match in size
- Compute weights for each class that are taken into account for the loss-function

Both were used and successfully counteracted the issue, however the classification performance was rather poor.

## Traits picked up by the classifier

The most "recently" implemented models tend to associate high event energy (i.e. energy of the neutrino) with "trackness", which of course is not the case. Also, it seems that the classifier misclassifies electron neutrino interactions (CC) for tracks. Even training specifically only samples of nue CC vs. numu CC did not yield any remedy to this issue.

## Training on "nice" tracks

The dataset was also filtered to discard tracks that appear cascade-like because of their low track length or too high cascade energy. The resulting filtered dataset only has very "obvious" and "nice" tracks. The idea is, that with this kind of dataset, the model should easier be able to pick up the "traits" of a track-shaped event. However, this did not turn out to improve the model performance on the overall dataset (while performance on nice-tracks improved).

## Exclude charge from the attributes of a vertex (DOM)

In order to prevent the model from learning the (wrong) association bertween high energy and "trackness", also charge was kept as a parameter from the model, which however did not result in any significant performance change.

## Filter values

The following values for the dataset filters were tried and used most of the time as reference:
- minimal track length: 70m / 75m
- maximal cascade energy: 10 GeV
- balancing the data classes: yes

## Performance

Training on the entire dataset (balancing the number of samples per class) as well as training electron neutrinos vs. muon neutrions (CC only) resulted in the best performance so far. Both models are able to achieve an accuracy of about 61% on the entire dataset. The number of layers as well as the hidden dimensionality did not provide any significant improvements so far.

# Using the Model on new data

A trained model can be used to classify real-world events (data). The `interface/icetray.py` was designed to include a method that takes an i3-frame as input and returns the trackness score. This however requires, that `icetray` as well as `torch` are run within the same python environment, which so far is not working properly. 

Alternatively, one can transform the new detector data into the same format that the training, testing and validation data uses. The wrapper script `evaluate.py`
allows to evaluate the model on any hd5 dataset, printing out the accuracy as well as mapping (filename, event id) -> prediction in a pickle file. A disadvantage of this method however is, that any data has to go through the entire dataset creation pipeline (see section 'Dataset') to fit the format required by the model. If no ground-truth class labels are at-hand, one has to make those up (the code expects there to be ground truth information), even though the evaluation code does not consider it at all.

# Future Experiments

## Standard RNNs

Neglecting the detector topology, one could input the position of a DOM as feature only and feed the hits as sequential vector data into an RNN.

Data has already been setup and needs to be checked and preprocessed, before a model can actually be trained. Other similar approaches on IceCube data promise some results at least.

## Graph Convolutional RNNs

Another, more sensible approach, that considers both the temporal evolution as well as the topology of the signal, would be to implement Graph Convolutional LSTMs as builing blocks of an RNN. This follows somewhat https://arxiv.org/pdf/1812.04206.pdf, without actually computing the graph Laplacians however (relying on an implementation of the Graph Convolution as above).

This way, a set of graphs (adding a new vertex for each trigger at a DOM) can be inputted to the GCRNN. That is, for each new pulse recorded by a DOM, the DOM is added to the graph is not present or the time and charge values are updated. The first "frame" of one event would be a single vertex graph, while the last "frame" would contain all vertices and show their latest hit time as well as the corresponding charge.

Creation of such a dataset is currently not implemented, and neither are models.



