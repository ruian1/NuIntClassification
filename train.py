import tensorflow as tf
import tensorflow.keras as keras
import os.path
from model import GCNN
import dataset
import numpy as np
import sys
tf.enable_eager_execution()

checkpoint_path = 'training/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    period = 5
)

#data = dataset.TestDataset('../test_data/data_centered.pkl')
#data = dataset.TestDataset('../test_data/test_data_big.pkl')
#data = dataset.TestDataset()
data = dataset.TestDataset(sys.argv[1])
hidden_dimensions_graph_convolutions, hidden_dimensions_fully_connected = eval(sys.argv[2])

#hidden_dimensions_graph_convolutions = [64, 64, 64]
#hidden_dimensions_fully_connected = [32, 16, 1]
model = GCNN(6, units_graph_convolutions = hidden_dimensions_graph_convolutions, 
    units_fully_connected = hidden_dimensions_fully_connected, dropout_rate=0.2, use_batchnorm=True)
optimizer = tf.keras.optimizers.Adam()

if hidden_dimensions_fully_connected[-1] == 1:
    loss = 'binary_crossentropy'
else:
    loss = 'sparse_categorical_crossentropy'

model.compile(optimizer=optimizer, 
              loss=loss,
              metrics=['accuracy'])


class LossCalback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        print(model.evaluate_generator(
            data.get_batches(batch_size=batch_size, train=False),
            steps=data.size(train=False) // batch_size))

batch_size = 128
model.fit_generator(
    data.get_batches(batch_size=batch_size, train=True), 
    steps_per_epoch = int(np.ceil(data.size(train=True) / batch_size)),
    epochs = 200,
    callbacks = [checkpoint_callback, LossCalback()],
    class_weight={0 : 551, 1 : 449}
    )
print(model.adjacency_layer.get_weights())