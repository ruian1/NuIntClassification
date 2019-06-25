import tensorflow as tf
from tensorflow import keras
import numpy as np
from .graph import *
from .util import *

tf.enable_eager_execution()

class RecurrentGraphConvolutionalNetwork(keras.Model):
    """ Model for a Recurrent, Graph Convolutional Network with dense graph structure. The graph is obtained using
    a Gaussian kernel on the pairwise node distances. """

    def __init__(self, num_input_features, units_graph_convolutions = [64, 64],
        units_lstm = [32, 16], units_fully_connected = [32, 1], dropout_rate=0.5, use_batchnorm=False, build_distances=False):
        """ Creates a GCNN model. 

        Parameters:
        -----------
        num_input_features : int
            Number of features for the input. 
        units_graph_convolutions : list
            The hidden units for each layer of graph convolution.
        units_lstm : list
            The hidden units for each LSTM layer.
        units_fully_connected : list
            The hidden units for each fully connected layer.
        dropout_rate : float
            Dropout rate.
        use_batchnorm : bool
            If batch normalization should be applied.
        build_distances : bool
            If the network is input with coordinates and has to build the pairwise distances itself.
        """
        super().__init__()
        self.is_binary_classifier = (units_graph_convolutions + units_lstm  + units_fully_connected)[-1] == 1
        self.adjacency_layer = GaussianAdjacencyMatrix(build_distances=build_distances)
        self.graph_convolutions, self.fully_connecteds, self.lstms = [], [], []

        # Add graph convolution blocks
        for layer_idx, hidden_dimension in enumerate(units_graph_convolutions):
            is_last_layer = layer_idx == len(units_graph_convolutions) - 1
            self.graph_convolutions.append(
                GraphConvolution(
                    hidden_dimension, 
                    dropout_rate = None if is_last_layer else dropout_rate,
                    use_activation = not is_last_layer,
                    use_batchnorm = False #not is_last_layer and use_batchnorm # TODO: implement padded batchnorm
                )
            )
        
        # Add LSTM blocks
        for layer_idx, hidden_dimension in enumerate(units_lstm):
            is_last_layer = layer_idx == len(units_lstm) - 1
            self.lstms.append(
                keras.layers.Bidirectional(keras.layers.CuDNNLSTM(hidden_dimension, return_sequences=not is_last_layer))
            )
        
        # Add fully connected blocks
        for layer_idx, hidden_dimension in enumerate(units_fully_connected):
            is_last_layer = layer_idx == len(units_fully_connected) - 1
            self.fully_connecteds.append(
                keras.layers.Dense(hidden_dimension, activation=None, use_bias=True)
            )
            if not is_last_layer:
                self.fully_connecteds.append(keras.layers.BatchNormalization())
            if not is_last_layer:
                self.fully_connecteds.append(keras.layers.ReLU())
            if not is_last_layer:
                self.fully_connecteds.append(
                    keras.layers.Dropout(dropout_rate)
                )

    def call(self, inputs):
        # Graph convolutions
        x, coordinates, masks = inputs
        A = self.adjacency_layer([coordinates, masks])
        for layer in self.graph_convolutions:
            x = layer([x, A, masks])
        # Average pooling of the node embeddings
        x = padded_vertex_mean(x, masks)
        # x = tf.reduce_sum(x, axis=1)
        # LSTMs
        for layer in self.lstms:
            x = layer(x)
        # Fully connected layers
        for layer in self.fully_connecteds:
            x = layer(x)
        # Output activation
        if self.is_binary_classifier:
            x = keras.activations.sigmoid(x)
        else:
            x = keras.activations.softmax(x)
        return x


    
