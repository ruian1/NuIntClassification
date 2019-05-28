import numpy as np
import tensorflow as tf
from model import RGCNN
import dataset

tf.enable_eager_execution()

"""
# Create test data
X = np.random.randn(1000, 40, 52, 8)
G = np.random.randn(1000, 40, 52, 3)
M = np.ones([1000, 40, 52, 52])
y = (np.random.randn(1000) < 0).astype(np.int)
X[y == 0, :, :, :] += 4
"""
class MyflatLayer(tf.keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, x):
        return tf.reduce_sum(x, axis=2)



"""
model = tf.keras.Sequential()
model.add(MyflatLayer())
model.add(tf.keras.layers.LSTM(32, return_sequences=False))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
"""

data = dataset.HD5Dataset('../data/data_dragon_sequential.hd5')


class LossLoggingCalback(tf.keras.callbacks.Callback):
    """ Callback for logging the losses at the end of each epoch. """

    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        print(model.evaluate_generator(
            data.get_batches(batch_size=batch_size, dataset='val'),
            steps=data.size(dataset='val') // batch_size))
        baseline_accuracy = data.get_baseline_accuracy(dataset='val')
        print(f'Baseline accuracy {baseline_accuracy}')

model = RGCNN(8, use_batchnorm=True)
model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

batch_size = 4

model.fit_generator(
        data.get_batches(batch_size=batch_size, dataset='train'), 
        steps_per_epoch = int(np.ceil(data.size(dataset='train') / batch_size)),
        epochs = 5,
        callbacks = [LossLoggingCalback()],
        class_weight = None
        )


print(model.fit([X, G, M], y))
