import numpy as np
import tensorflow as tf
from dataset import HD5Dataset
from model import *

tf.enable_eager_execution()

class LossLoggingCalback(tf.keras.callbacks.Callback):
    """ Callback for logging the losses at the end of each epoch. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_history = []

    def on_epoch_end(self, epoch, logs=None):
        def generator():
            for (X, _, _), y in data.get_batches(batch_size=batch_size, dataset='val'):
                yield X.sum(1), y

        print(model.evaluate_generator(
            generator(),
            steps=int(np.ceil(data.size(dataset='val') // batch_size))))
        y_pred = model.predict_generator(generator(),
                       steps=int(np.ceil(data.size(dataset='val') / batch_size)))
        print(np.unique(y_pred), np.unique(y_pred).shape)

        baseline_accuracy = data.get_baseline_accuracy(dataset='val')
        
        print(f'Baseline accuracy {baseline_accuracy}')

"""
data = HD5Dataset('../data/data_dragon3.hd5', features=['ChargeFirstPulse', 'TimeFirstPulse', 'ChargeLastPulse', 'TimeLastPulse',
    'IntegratedCharge', 'ChargeMaxPulse', 'TimeMaxPulse', 'TimeVariance', 'VertexX', 'VertexY', 'VertexZ'])
batch_size = 64
def data_generator():
    for (X, C, M), y in data.get_batches(batch_size=batch_size):
        yield X.sum(1), y

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
"""
"""
X = np.random.randn(1000, 100, 10, 3)
A = np.ones([1000, 100, 10, 10])
M = np.ones([1000, 100, 10, 10])
y = (np.random.randn(1000) > 0).astype(np.int)
X[y > 0] = np.abs(X[y > 0])
X[y <= 0] = -np.abs(X[y <= 0])

"""
"""
model = tf.keras.Sequential(layers=[
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=False)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
"""
"""

model = RecurrentGraphConvolutionalNetwork(3, units_graph_convolutions=[64, 64], units_fully_connected=[16, 1], units_lstm=[16, 16])
"""

n = 10000

X = np.random.randn(n, 25, 5)
F = np.random.randn(n, 10)
A = np.random.randn(n, 25, 25)
M = np.zeros([n, 25, 25])
M[:, :15, :15] = 1
y = (np.random.randn(n) > 0).astype(np.int)
X[y > 0] = np.abs(X[y > 0])
X[y <= 0] = -np.abs(X[y <= 0])
F[y > 0] = np.abs(F[y > 0])
F[y < 0] = -np.abs(F[y < 0])
model = GraphConvolutionalNetwork(5, units_graph_features=[64, 64], use_batchnorm=False)

optimizer = tf.keras.optimizers.Adam(lr=1e-4)
loss = 'binary_crossentropy'
model.compile(optimizer=optimizer, 
            loss=loss,
            metrics=['accuracy'])
"""
model.fit_generator(
        data_generator(), 
        steps_per_epoch = int(np.ceil(data.size(dataset='train') / batch_size)),
        epochs = 50,
        callbacks = [LossLoggingCalback()],
        class_weight = data.get_class_weights()
        )
"""
model.fit(x=[X, F, A, M], y=y, epochs=20, validation_split=0.1)
#model.fit(x=X, y=y, epochs=20, batch_size=32)