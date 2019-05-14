import tensorflow as tf
import tensorflow.keras as keras
import os.path
from model import GCNN
import data
import numpy as np

checkpoint_path = 'training/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    period = 5
)



"""
n_test = 5000
X = np.random.randn(n_test, 13, 5)
X[:n_test // 2] += np.random.rand(n_test // 2, 13, 5)
C = np.random.rand(n_test, 13, 3)
y = np.zeros(n_test)
y[:n_test // 2] = 1
data_numu = np.load('../test_data/numu.npz', allow_pickle=True, encoding='bytes')
data_nue = np.load('../test_data/nue.npz', allow_pickle=True, encoding='bytes')
data_nutau = np.load('../test_data/nutau.npz', allow_pickle=True, encoding='bytes')
data = [data_numu, data_nue, data_nutau]

features, coordinates, targets = [], [], []
for class_idx, dataset in enumerate(data):
    features.append(dataset['features'])
    coordinates.append(dataset['coordinates'])
    targets.append(np.zeros(len(features[-1])) + class_idx)

X = np.concatenate(features, axis=0)
y = np.concatenate(targets, axis=0)
C = np.concatenate(coordinates, axis=0)

"""

n_graphs_per_class = 3000
X, C, masks, y = data.load_padded_test_data(n_graphs_per_class)



model = GCNN(6, [64, 64, 64])
"""
model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(5,)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax),
])
print(model.summary)
"""
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(X.shape, C.shape, y.shape)

model.fit([X, C, masks], y,
    epochs = 5, callbacks = [checkpoint_callback], validation_split=0.2
    )