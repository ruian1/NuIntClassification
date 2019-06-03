import numpy as np
import tensorflow as tf
from dataset import HD5Dataset
from model import RecurrentGraphConvolutionalNetwork

tf.enable_eager_execution()

X = np.random.randn(1000, 100, 10, 3)
A = np.ones([1000, 100, 10, 10])
M = np.ones([1000, 100, 10, 10])
y = (np.random.randn(1000) > 0).astype(np.int)
X[y > 0] = np.abs(X[y > 0])
X[y <= 0] = -np.abs(X[y <= 0])

"""
model = tf.keras.Sequential(layers=[
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=False)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
"""

model = RecurrentGraphConvolutionalNetwork(3, units_graph_convolutions=[64, 64], units_fully_connected=[16, 1], units_lstm=[16, 16])
optimizer = tf.keras.optimizers.Adam()
loss = 'binary_crossentropy'
model.compile(optimizer=optimizer, 
            loss=loss,
            metrics=['accuracy'])
model.fit(x=[X, A, M], y=y, epochs=20)
#model.fit(x=X, y=y, epochs=20, batch_size=32)