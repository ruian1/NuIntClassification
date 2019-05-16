import tensorflow as tf
import tensorflow.keras as keras
import os.path
from model import GCNN
import dataset
import numpy as np
tf.enable_eager_execution()

checkpoint_path = 'training/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    period = 5
)

data = dataset.TestDataset('../test_data/test_data_big.pkl')



model = GCNN(6, ([64, 64, 64], [32, 16, 2]))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


class LossCalback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        print(model.evaluate_generator(
            data.get_batches(batch_size=batch_size, train=False),
            steps=data.size(train=False) // batch_size))

batch_size = 64
model.fit_generator(
    data.get_batches(batch_size=batch_size, train=True), 
    steps_per_epoch = int(np.ceil(data.size(train=True) / batch_size)),
    epochs = 100,
    callbacks = [checkpoint_callback, LossCalback()],
    class_weight={0 : 551, 1 : 449}
    )
print(model.adjacency_layer.get_weights())