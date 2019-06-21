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
        A = padded_softmax(A, masks)
        #A = tf.Print(A, [A], 'Adjacency', summarize=100000000)
        #A = tf.print()
        #A = tf.Print(A, [A], 'Adjacency', summarize=100000000)
        return A



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

    def call(self, inputs, training=None):
        x, A, masks = inputs
        hidden = self.dense(x)
        a = tf.matmul(A, hidden)
        #a = tf.concat([a, x], axis=-1)
        if self.use_batchnorm:
            a = self.bn([a, masks], training=training)
        if self.use_activation:
            a = self.activation(a)
            #x = tf.concat([x, activated], axis=-1)
        #a = tf.concat([a, x], axis=-1)
        if self.dropout_rate:
            a = self.dropout(a, training=training)
        return a

    
class BatchNormalization(keras.layers.Layer):
    """ Layer that applies feature and batch normalization accounting for padded zeros. """

    def __init__(self, *args, momentum=0.99, **kwargs):
        """ Batchnorm layer that accounts for padded vertices.
        
        Parameters:
        -----------
        momentum : float
            The momentum for the moving mean and variance.
        """
        super().__init__(*args, **kwargs)
        self.momentum = momentum

    def build(self, input_shape):
        shape_X, _ = input_shape
        weight_shape = shape_X.as_list()
        for idx in range(len(weight_shape) - 1):
            weight_shape[idx] = 1
        self.beta = self.add_weight('beta', shape = weight_shape)
        self.gamma = self.add_weight('gamma', shape = weight_shape)
        self.moving_mean = self.add_weight('moving_mean', shape=weight_shape, trainable=False)
        self.moving_variance = self.add_weight('moving_variance', shape=weight_shape, trainable=False)

    def call(self, inputs, training=None):
        X, masks = inputs # X is of shape [num_samples, num_vertices, num_features]
        #X = tf.Print(X, [X], 'X')
        if training: # This is super weird, that during fit
            # Normalize over batch and vertices
            vertex_mean = tf.expand_dims(padded_vertex_mean(X, masks), -2)
            batch_mean = tf.reduce_mean(vertex_mean, axis=0, keepdims=True)
            X_centered = X - batch_mean
            batch_variance = tf.expand_dims(padded_vertex_mean(X_centered ** 2, masks), -2)
            batch_variance = tf.reduce_mean(batch_variance, axis=0, keepdims=True)
            X_normalized = X_centered / (tf.sqrt(batch_variance) + 1e-20)
            # Update the moving mean and variance
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * batch_mean
            self.moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * batch_variance
        else:
            # Use the moving mean and variance to scale the batch
            #self.moving_mean = tf.Print(self.moving_mean, [self.moving_mean], 'mv')
            X_normalized = X - self.moving_mean / (tf.sqrt(self.moving_variance) + 1e-20)

        return self.gamma * X_normalized + self.beta
