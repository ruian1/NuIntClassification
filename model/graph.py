import tensorflow as tf
from tensorflow import keras
import numpy as np
from .util import *
tf.enable_eager_execution()

class GaussianAdjacencyMatrix(keras.layers.Layer):
    """ Layer that creates the adjacency matrix based on a set of euclidean coordinates. """

    def __init__(self, build_distances=True, **kwargs):
        """ Creates a layer that builds a gaussian adjacency matrix with learnable sigma.
        
        Parameters:
        -----------
        build_distances : bool
            If true, the layer receives coordinates as inputs and builds pairwise distances.
            If false, the layer receives precomputed pairwise distances.
        """
        super().__init__(**kwargs)
        self.build_distances = build_distances

    def build(self, input_shape):
        shape_coordinates, _ = input_shape
        self.permutation = [axis for axis in range(len(shape_coordinates.as_list()))]
        self.permutation[-1], self.permutation[-2] = self.permutation[-2], self.permutation[-1]
        self.sigma = self.add_weight('sigma', shape=[1]) #, constraint=keras.constraints.NonNeg()) #
        
    def call(self, inputs):
        """ Creates a gaussian adjacency matrix. 
        
        Parameters:
        -----------
        inputs : tuple
            If build_distances == True, then inputs consists of
            - coordinates : tf.tensor, shape [..., N, 3]
            - masks : tf.tensor, shape [..., N, N]
            else it consists of:
            - distances : tf.tensor, shape [..., N, N]
            - masks : tf.tensor, shape [..., N, N]
        
        Returns:
        --------
        A : tf.tensor, shape [..., N, N]
            The adjacency matrix of the graph.
        """
        distances, masks = inputs
        # Apply a gaussian kernel and normalize using a softmax
        A = tf.exp(-distances / (self.sigma ** 2))
        return padded_softmax(A, masks)



class GraphConvolution(keras.layers.Layer):
    """ Block that implements a graph convolution. """

    def __init__(self, hidden_dim, dropout_rate=None, use_activation=True, use_batchnorm=True, input_shape=None, **kwargs):
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

    def build(self, input_shape):
        self.dense = keras.layers.Dense(self.hidden_dim, input_shape=input_shape, use_bias=True)
        if self.use_batchnorm:
            self.bn = BatchNormalization()
        if self.use_activation:
            self.activation = keras.layers.ReLU()
        if self.dropout_rate:
            self.dropout = keras.layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs):
        inputs, A, masks = inputs
        x = tf.matmul(A, inputs)
        x = tf.concat([x, inputs], axis=-1)
        x = self.dense(x)
        if self.use_batchnorm:
            x = self.bn([x, masks])
        if self.use_activation:
            activated = self.activation(x)
            #x = tf.concat([x, activated], axis=-1)
        if self.dropout_rate:
            x = self.dropout(x)
        return x

    
class BatchNormalization(keras.layers.Layer):
    """ Layer that applies feature and batch normalization accounting for padded zeros. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        shape_X, _ = input_shape
        weight_shape = shape_X.as_list()
        for idx in range(len(weight_shape) - 1):
            weight_shape[idx] = 1
        self.beta = self.add_weight('beta', shape = weight_shape)
        self.gamma = self.add_weight('gamma', shape = weight_shape)

    def call(self, inputs):
        X, masks = inputs # X is of shape [num_samples, num_vertices, num_features]
        # Normalize over batch and vertices
        vertex_mean = tf.expand_dims(padded_vertex_mean(X, masks), -2)
        batch_mean = tf.reduce_mean(vertex_mean, axis=0, keepdims=True)
        X_centered = X - batch_mean
        X_var = tf.expand_dims(padded_vertex_mean(X_centered ** 2, masks), -2)
        X_var = tf.reduce_mean(X_var, axis=0, keepdims=True)
        X_normalized = X_centered / tf.sqrt(X_var) + 1e-20
        return self.gamma * X_normalized + self.beta
