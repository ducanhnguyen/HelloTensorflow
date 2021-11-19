"""
Save the information of the model during training
"""

import tensorflow as tf
from keras.callbacks import CSVLogger

# load data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# ModelCheckpoint
model.fit(training_images, training_labels,
          epochs=5,
          verbose=True,
          validation_data=(test_images, test_labels),
          callbacks=[CSVLogger('../log.csv'),
                     ])
