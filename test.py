import numpy as np
import tensorflow as tf
from model import RGCNN

tf.enable_eager_execution()

# Create test data
X = np.random.randn(1000, 40, 52, 8)
G = np.random.randn(1000, 40, 52, 3)
M = np.ones([1000, 40, 52, 52])
y = (np.random.randn(1000) < 0).astype(np.int)
X[y == 0, :, :, :] += 4

class MyflatLayer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def call(x):
        x_shape = tf.shape(x)
        x = tf.reshape(x, [-1, x_shape[1], tf.reduce_prod(x_shape[2:])])
        return x

model = tf.keras.Sequential()
model.add(MyflatLayer)
model.add(tf.keras.layers.LSTM(32, return_sequences=False))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


#model = RGCNN(8, use_batchnorm=True)
model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')
print(model.fit(X, y))
