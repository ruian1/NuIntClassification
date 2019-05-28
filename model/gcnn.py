import tensorflow as tf
from tensorflow import keras
import numpy as np
tf.enable_eager_execution()

class GCNN(keras.Model):
    """ Model for a Graph Convolutional Network with dense graph structure. The graph is obtained using
    a Gaussian kernel on the pairwise node distances. """

    def __init__(self, num_input_features, units_graph_convolutions = [64, 64], units_fully_connected = [32, 1], 
        dropout_rate=0.5, use_batchnorm=True):
        """ Creates a GCNN model. 

        Parameters:
        -----------
        num_input_features : int
            Number of features for the input. 
        units_graph_convolutions : list
            The hidden units for each layer of graph convolution.
        units_fully_connected : list
            The hidden units for each fully connected layer.
        dropout_rate : float
            Dropout rate.
        use_batchnorm : bool
            If batch normalization should be applied.
        """
        super().__init__()
        self.is_binary_classifier = (units_graph_convolutions + units_fully_connected)[-1] == 1
        self.adjacency_layer = AdjacencyMatrixLayer()
        self.graph_convolutions, self.fully_connecteds = [], []

        # Add graph convolution blocks
        for layer_idx, hidden_dimension in enumerate(units_graph_convolutions):
            is_last_layer = layer_idx == len(units_graph_convolutions) - 1
            self.graph_convolutions.append(
                GCNNBlock(
                    hidden_dimension, 
                    dropout_rate = None if is_last_layer else dropout_rate,
                    use_activation = not is_last_layer,
                    use_batchnorm = False#not is_last_layer and use_batchnorm # TODO: implement padded batchnorm
                )
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
        # x = padded_vertex_mean(x, masks)
        x = tf.reduce_sum(x, axis=1)
        # Fully connected layers
        for layer in self.fully_connecteds:
            x = layer(x)
        # Output activation
        if self.is_binary_classifier:
            x = keras.activations.sigmoid(x)
        else:
            x = keras.activations.softmax(x)
        return x


class GCNNBlock(keras.layers.Layer):
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
        #self.bias = self.add_weight('bias', shape=[1])

    def build(self, input_shape):
        self.dense = keras.layers.Dense(self.hidden_dim, input_shape=input_shape, use_bias=True)
        if self.use_batchnorm:
            self.bn = FeatureNormalization()
        if self.use_activation:
            self.activation = keras.layers.ReLU()
        if self.dropout_rate:
            self.dropout = keras.layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs):
        inputs, A, masks = inputs
        x = tf.matmul(A, inputs)
        x = tf.concat([x, inputs], axis=2)
        x = self.dense(x)
        #x += self.bias
        if self.use_batchnorm:
            x = self.bn([x, masks])
        if self.use_activation:
            activated = self.activation(x)
            x = tf.concat([x, activated], axis=2)
        if self.dropout_rate:
            x = self.dropout(x)
        return x

class AdjacencyMatrixLayer(keras.layers.Layer):
    """ Layer that creates the adjacency matrix based on a set of euclidean coordinates. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.sigma = self.add_weight('sigma', shape=[1]) #
    
    def call(self, inputs):
        # Create a pairwise distance matrix D, where D[i, j] = |c[i] - c[j]|_{L2}^2
        # Using the identity: D[i, j] = (c[i] - c[j])(c[i] - c[j])^T = r[i] - 2 c[i]c[j]^T + r[j]
        # where r[i] is the squared L2 norm of the i-th coordinate
        coordinates, masks = inputs
        coordinate_norms = tf.reduce_sum(coordinates ** 2, 2, keepdims=True)
        distances = coordinate_norms - 2 * tf.matmul(coordinates, tf.transpose(coordinates, perm=[0, 2, 1])) + tf.transpose(coordinate_norms, perm=[0, 2, 1])
        # Apply a gaussian kernel and normalize using a softmax
        A = tf.exp(-distances / (self.sigma ** 2))
        return padded_softmax(A, masks)


class FeatureNormalization(keras.layers.Layer):
    """ Layer that applies feature normalization accounting for padded zeros. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # TODO: add gamma and mu parameters for learnable centering and scaling
        shape_X, _ = input_shape
        self.beta = self.add_weight('beta', shape=[1, 1, shape_X.as_list()[2]])
        self.gamma = self.add_weight('gamma', shape=[1, 1, shape_X.as_list()[2]])

    def call(self, inputs):
        X, masks = inputs # X is of shape [num_samples, num_vertices, num_features]
        X_mean = tf.expand_dims(padded_vertex_mean(X, masks), 1)
        X_centered = X - X_mean
        X_var = padded_vertex_mean(X_centered ** 2, masks)
        X_normalized = X_centered / tf.expand_dims(tf.sqrt(X_var) + 1e-20, 1)
        return self.gamma * X_normalized + self.beta


def padded_vertex_mean(X, masks):
    """ Calculates the mean over all vertices and consideres padded vertices. 
    
    Parameters:
    -----------
    X : tf.tensor, shape [num_batches, num_vertices, D]
        The tensor to calculate the mean over all vertices (axis 1).
    masks : tf.tensor, shape [num_batches, num_vertices, num_vertices]
        Masks for adjacency matrix of the graphs.

    Returns:
    --------
    X_mean : tf.tensor, shape [num_batches, D]
        Mean over all non-padded vertices.
    """
    vertex_masks = tf.reduce_max(masks, axis=[-1]) # Masking each vertex individually 
    num_vertices = tf.reduce_sum(vertex_masks, axis=1, keepdims=True) # Number of vertices per sample
    X_mean = tf.reduce_sum(X, axis=1)
    X_mean /= num_vertices + 1e-20
    return X_mean
    
def padded_softmax(A, masks):
    """ Calculates the softmax along axis 2 considering padded vertices. 
    
    Parameters:
    -----------
    A : tf.tensor, shape [num_samples, num_vertices, num_vertices]
        The adjacency matrices.
    masks : tf.tensor, shape [num_samples, num_vertices, num_vertices]
        The masks for the adjacency matrices.

    Returns:
    --------
    A : tf.tensor, shape [num_samples, num_vertices, num_vertices]  
        The masked adjacency matrices normalized using a softmax while considering padded vertices.
    """
    A = tf.nn.softmax(A, axis=2)
    A *= masks
    normalization = tf.reduce_sum(A, axis=2, keepdims=True) + 1e-20
    return A / normalization


    
