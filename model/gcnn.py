import tensorflow as tf
from tensorflow import keras
import numpy as np
from .graph import *
from .util import *
tf.enable_eager_execution()

class GraphConvolutionalNetwork(keras.Model):
    """ Model for a Graph Convolutional Network with dense graph structure. The graph is obtained using
    a Gaussian kernel on the pairwise node distances. """

    def __init__(self, num_input_features, units_graph_convolutions = [64, 64], units_fully_connected = [32, 1], 
        units_graph_features=None, dropout_rate=0.5, use_batchnorm=True, build_distances=False):
        """ Creates a GCNN model. 

        Parameters:
        -----------
        num_input_features : int
            Number of features for the input. 
        units_graph_convolutions : list
            The hidden units for each layer of graph convolution.
        units_fully_connected : list
            The hidden units for each fully connected layer.
        units_graph_features : list or None
            If a list, the network expects as an input X, F, coordinates / distances, masks, where F
            represents a feature matrix that contains features for the entire graph. These features
            are processed by a seperate feed forward layer structure and is aggregated with the
            graph representation that is obtained by pooling the node embeddings after all GC layers.
        dropout_rate : float
            Dropout rate.
        use_batchnorm : bool
            If batch normalization should be applied.
        build_distances : bool
            If the network is input with coordinates and has to build the pairwise distances itself.
        """
        super().__init__()
        self.is_binary_classifier = (units_graph_convolutions + units_fully_connected)[-1] == 1
        self.adjacency_layer = GaussianAdjacencyMatrix(build_distances=build_distances)
        self.graph_convolutions, self.fully_connecteds = [], []
        self.number_classes = (units_graph_convolutions + units_fully_connected)[-1]

        # Add graph convolution blocks
        for layer_idx, hidden_dimension in enumerate(units_graph_convolutions):
            is_last_layer = layer_idx == len(units_graph_convolutions) - 1
            self.graph_convolutions.append(
                GraphConvolution(
                    hidden_dimension, 
                    dropout_rate = None if is_last_layer else dropout_rate,
                    use_activation = not is_last_layer,
                    use_batchnorm = False #use_batchnorm and not is_last_layer,
                )
            )
        
        # Add fully connected blocks
        for layer_idx, hidden_dimension in enumerate(units_fully_connected):
            is_last_layer = layer_idx == len(units_fully_connected) - 1
            self.fully_connecteds.append(
                keras.layers.Dense(hidden_dimension, activation=None, use_bias=True)
            )
            if not is_last_layer and use_batchnorm:
                self.fully_connecteds.append(keras.layers.BatchNormalization())
            if not is_last_layer:
                self.fully_connecteds.append(keras.layers.ReLU())
            if not is_last_layer:
                self.fully_connecteds.append(keras.layers.Dropout(dropout_rate))

        # Add graph feature layers
        if units_graph_features is not None:
            self.graph_feature_layers = []
            for layer_idx, hidden_dimension in enumerate(units_graph_features):
                is_last_layer = layer_idx == len(units_graph_features) - 1
                self.graph_feature_layers.append(
                    keras.layers.Dense(hidden_dimension, activation=None, use_bias=True)
                )
                if not is_last_layer:
                    self.graph_feature_layers.append(keras.layers.BatchNormalization())
                self.graph_feature_layers.append(keras.layers.ReLU())
                if not is_last_layer:
                    self.graph_feature_layers.append(keras.layers.Dropout(dropout_rate))
        else:
            self.graph_feature_layers = None


    def call(self, inputs):
        # Graph convolutions
        if self.graph_feature_layers is None:
            x, coordinates, masks = inputs
        else:
            x, graph_features, coordinates, masks = inputs
        A = self.adjacency_layer([coordinates, masks])
        for layer in self.graph_convolutions:
            x = layer([x, A, masks])
        # Average pooling of the node embeddings
        x = padded_vertex_mean(x, masks)

        # Run the graph feature NN
        if self.graph_feature_layers is not None:
            for layer in self.graph_feature_layers:
                graph_features = layer(graph_features)
            x = tf.concat([x, graph_features], -1)

        # Fully connected layers
        for layer in self.fully_connecteds:
            x = layer(x)
        # Output activation
        if self.is_binary_classifier:
            x = keras.activations.sigmoid(x)
        else:
            x = keras.activations.softmax(x)
        return x

    def get_num_classes(self):
        """ Returns the number of classes the model predicts. 
        
        Returns:
        --------
        num_classes : int
            The numer of classes / dimensionality of the last layer.
        """
        return self.number_classes




    
