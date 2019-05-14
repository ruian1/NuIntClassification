import tensorflow as tf
from tensorflow import keras
import numpy as np


class GCNN(keras.Model):
    """ Model for a Graph Convolutional Network with dense graph structure. The graph is obtained using
    a Gaussian kernel on the pairwise node distances. """

    def __init__(self, num_input_features, hidden_dimensions, num_classes, dropout_rate=0.5, use_batchnorm=True):
        """ Creates a GCNN model. 

        Parameters:
        -----------
        num_input_features : int
            Number of features for the input. 
        hidden_dimensions : list
            A list of ints, representing the hidden dimensionalities of the filter weights.
        num_classes : int
            The number of classes. A graph convolution layer with according dimensionality and
            a softmax are put the end of the computational pipeline of the model.
        dropout_rate : float
            Dropout rate.
        use_batchnorm : bool
            If batch normalization should be applied.
        """
        super().__init__()
        self.adjacency_layer = AdjacencyMatrixLayer()
        self.blocks = [
            GCNNBlock(hidden_dimension, dropout_rate=dropout_rate, use_activation=True, use_batchnorm=True)
            for i, hidden_dimension in enumerate(hidden_dimensions)
        ]
        self.blocks.append(
            GCNNBlock(num_classes, dropout_rate=None, use_activation=False, use_batchnorm=False)
        )
        self.softmax = keras.layers.Softmax(axis=1)

    def call(self, inputs):
        x, coordinates = inputs
        A = self.adjacency_layer(coordinates)
        for layer in self.blocks:
            x = layer([x, A])
        # Average pooling of the node embeddings
        x = tf.reduce_mean(x, axis=1)
        return self.softmax(x)


class GCNNBlock(keras.layers.Layer):
    """ Block that implements a graph convolution. """

    def __init__(self, hidden_dim, dropout_rate=None, use_activation=True, use_batchnorm=True, **kwargs):
        """ Initializes the GCNN Layer. 
        Parameters:
        -----------
        hidden_dim : int or None
            Kernel size, i.e. hidden dimensionality.
        dropout_rate : None or float
            The dropout rate after the activation is applied. If None, no dropout layer is added to the block.
        use_activation : bool
            If true, applies a ReLU activation after the linear operation.
        use_batchnorm : bool
            If true, applies batch normalization after the linear operation.
        """
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_activation = use_activation
        self.use_batchnorm = use_batchnorm
        self.bias = self.add_weight('bias', shape=[1])

    def build(self, input_shape):
        self.dense = keras.layers.Dense(self.hidden_dim, input_shape=input_shape, use_bias=False)
        if self.use_batchnorm:
            self.bn = keras.layers.BatchNormalization()
        if self.use_activation:
            self.activation = keras.layers.ReLU()
        if self.dropout_rate:
            self.dropout = keras.layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs):
        inputs, A = inputs
        x = tf.matmul(A, inputs)
        x = tf.concat([x, inputs], 2)
        x = self.dense(x)
        x += self.bias
        if self.use_batchnorm:
            x = self.bn(x)
        if self.use_activation:
            activated = self.activation(x)
            x = tf.concat([x, activated], 2)
        if self.dropout_rate:
            x = self.dropout(x)
        return x

class AdjacencyMatrixLayer(keras.layers.Layer):
    """ Layer that creates the adjacency matrix based on a set of euclidean coordinates. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.precision_squared = self.add_weight('precision_squared', shape=[1]) # sigma^{-2}
    
    def call(self, coordinates):
        # Create a pairwise distance matrix D, where D[i, j] = |c[i] - c[j]|_{L2}^2
        # Using the identity: D[i, j] = (c[i] - c[j])(c[i] - c[j])^T = r[i] - 2 c[i]c[j]^T + r[j]
        # where r[i] is the squared L2 norm of the i-th coordinate
        A = tf.map_fn(self.build_adjacency_matrix_from_coordinates, coordinates)
        return A


    def build_adjacency_matrix_from_coordinates(self, coordinates):
        """ Creates the adjacency matrix of a graph based on a set of coordinates.
        
        Parameters:
        -----------
        coordiantes : tf.tensor, shape [K, 3]
            The coordinates that will be used to construct the dense graph.
        
        Returns:
        --------
        A : tf.tensor, shape [K, K]
            The adjacency matrix.
        """
        coordinate_norms = tf.reduce_sum(coordinates * coordinates, 1)
        coordinate_norms = tf.reshape(coordinate_norms, [-1, 1])
        distances = coordinate_norms - 2 * tf.matmul(coordinates, tf.transpose(coordinates)) + tf.transpose(coordinate_norms)
        # Apply a gaussian kernel and normalize using a softmax
        A = tf.exp(-distances * self.precision_squared)
        A = tf.nn.softmax(A, axis=1)
        return A



        

    