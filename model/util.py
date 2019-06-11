import tensorflow as tf
from tensorflow import keras
import numpy as np
tf.enable_eager_execution()

def padded_vertex_mean(X, masks):
    """ Calculates the mean over all vertices and consideres padded vertices. 
    
    Parameters:
    -----------
    X : tf.tensor, shape [K1, ..., num_vertices, num_features]
        The tensor to calculate the mean over all vertices (axis -2).
    masks : tf.tensor, shape [K1, ..., num_vertices, num_vertices]
        Masks for adjacency matrix of the graphs.

    Returns:
    --------
    X_mean : tf.tensor, shape [K1, ..., num_features]
        Mean over all non-padded vertices.
    """
    vertex_masks = tf.reduce_max(masks, axis=[-1]) # Masking each vertex individually 
    num_vertices = tf.reduce_sum(vertex_masks, axis=-1, keepdims=True) # Number of vertices per sample
    X_mean = tf.reduce_sum(X, axis=-2)
    X_mean /= num_vertices + 1e-20
    return X_mean

def padded_softmax(A, masks, axis=-1):
    """ Calculates the softmax along an axis considering padded vertices. 
    
    Parameters:
    -----------
    A : tf.tensor, shape [K1, ..., num_vertices, num_vertices]
        The adjacency matrices.
    masks : tf.tensor, shape [K1, ..., num_vertices, num_vertices]
        The masks for the adjacency matrices.
    axis : int
        The axis along which to perform a softmax (usually the features)

    Returns:
    --------
    A : tf.tensor, shape [num_samples, num_vertices, num_vertices]  
        The masked adjacency matrices normalized using a softmax while considering padded vertices.
    """
    A = tf.nn.softmax(A, axis=axis)
    A *= masks
    normalization = tf.reduce_sum(A, axis=axis, keepdims=True) + 1e-20
    return A / normalization
