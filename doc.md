# Documentation of the code and conducted experiements

# Dataset

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


