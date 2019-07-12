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

All i3 files are aggregated into a single large hdf5 file that contains all events that are available. The python scripts `create_dataset/create_dataset.py` and `create_dataset/create_recurrent_dataset.py` are responsible for parsing a single i3 file each and storing the ouput into an hdf5 file. Since parsing an i3 file and handling its content requires the `icetray` package, it has to be run withing a singularity environment. The slurm script as well as a shell script to wrap this process are designed to operate on computecanada's `cedar` cluster.

Jobs for creating datasets (`create_dataset/create_dataset_slurm_job.sh` and `create_dataset/create_recurrent_dataset_slurm_job.sh`) are supposed to be run as array-jobs, where the number of jobs should match the number of i3 files to be parsed.

The script `create_dataset/concatenate_datasets.py` gets rid of all metadata and creates one single large hdf5 datafile, which contains all events.

In order to speed up the training, the dataset is split into training, validation and testing data beforehand. 10% of the entire data are used for validation, another 10% for testing, such that 80% of all events remain for training. The splitting is performed using `create_dataset/transform_dataset.py`. This script also shuffles the data.

## DOM Coordinates

The coordinates of each DOM are normalized, before they are fed into the GCN. They are centered to the mean of the entire event. This mean can either be a weighted mean (by charges) or just an unweighted one. An empircally obtained value is used to scale the coordinates along each axis, such that they approximately fit into a distribution, that somewhat resembles the standard normal distribution.

## Pulse Times

Another way to incorporate the reconstruction is, to calculate physically when a DOM is expected to recognize charge first, assuming the length and direction of the reconstructed track are correct. It is then possible, to calculate a difference between the time of actually observing any light and this expected time value. The idea is, that for tracks, the difference should not be too large, while for cascade-like events, there is a huge discrepancy to expect. These attributes are also available as features in the hdf5 file, as the scripts for creating datasets calculate these values as well.

A few remarks regarding this calculation: Since 



The dataset is generated in multiple steps. The original source is an i3 file that contains detector data. From this i3 file a vertex attributed graph is generated for each event. In these events a DOM represents a vertex, while the graph itself is densly connected, meaning that each DOM is connected to each other DOM via an edge, the strength of which corresponds to the spatial distance between the DOMs.

### Vertex Features

For each DOM the following attributes are extracted and used to train the network:
- Charge of the first, last and maximal pulse
- Time of the first, last and maximal (w.r.t. charge) pulse
- Time Difference to expectation using the reconstruction of the first, last and maximal (w.r.t. charge) pulse
- Standard Devation of Pulse Times
- Coordinates of the DOM (x, y, z)

Time and charge values are scaled to approximately fit a [0, 1] range.

For time values a second set is generated in the following way: Using the track reconstruction (i.e. assuming it was true), one can calculate the time when a DOM is expected to register Cherenkov light. Due to the fact that the Level 6 reconstruction used usually contains a track which is fully enclosed by the detector, many DOMs which actually would see Cherenkov light due to scattering are associated with no value for the expected time. To counteract this issue, the track length is increased to infity, such that estimates for the Cherenkov time (not accounting for scattering obviously) can be obtained at least. Only DOMs, for which the point on the reconstructed track is before the actual interaction vertex, are assigned with a NaN time.
Also the expected time of light caused by the interaction itself (originating at the interaction vertex) is considered, and which ever is the smaller one is set as the expected time. The features of the dataset contain a difference of the actually observed time at a DOM and the expected time using the reconstruction (cascade and track).

The coordinates are centered arround their mean (possibly charge weighted) and scaled by an empircal value of 50m to resemble an approximate standard devation of ~1. Note that the same transformation is applied to the reconstruction.

### Graph Features

For each event the reconstruction is extracted as a graph-wise feature. These include the position (x, y, z) and angles (zenith and azimuth). Coordinates are adjusted to fit the same system as the coordinates of the DOMs.

## Splitting the data

In order to optimize training times, the data is shuffeld and split beforehand, such that the training indices of the samples always are sequential.

## Memory Maps

When training a model, the entire data is loaded as a NumPy memory map. This memory map allows efficient access to the data values and significantly increases training times (from hours to minutes per epoch). Memory maps are also cached in ./memmaps, which decreases the setup time of a model by a large amount.

# Training

Training is done using a json configuration that defines the data and the model architecture and hyperparameters as well as learning parameters such as the learning rate and scheduler. There exists a default configuration, which is overriden by any configuration file.

For each experiment, a model id is generated automatically and results can be refered to by this model id. Performance on the validation as well as testing data, together with model parameters after each epoch are saved.

# Experiments conducted

Training the model on the entire dataset resulted in poor performance. Thus, different training setups were considered.

1. Training on energies with long tracks and low cascade energies (track length >= 70m, cascade energy < 10 GeV)

2. Training NuMu CC vs. Nue CC: The model seems to identify Nue CC as tracks as well, which is why this setup was used

None of the setups above however result in any significant improvements.

## Model evaluation

The jupyter notebook ```baselines.ipynb``` evaluates any model with respect to the testing data (filtered w.r.t. to track length, cascade_energy, etc. ) as well as unfiltered samples. It also creates plots for which features are associated with "trackness" in the original detector data.

# Future Experiments

## Graph Convolutional RNNs

Using the dataset as described above, somewhat neglects temporal information. Thus, one could implement LSTM cells which take graph shaped inputs.

## Standard RNNs

Neglecting the detector topology, one could input the position of a DOM as feature only and feed the hits as sequential vector data into an RNN.


