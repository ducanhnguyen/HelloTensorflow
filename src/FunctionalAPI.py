"""
Define a model by using Functional API
"""

import tensorflow as tf
from tensorflow.keras.models import Model

# Create a model using functional API
input_layer = tf.keras.Input(shape=(28, 28))
flatten_layer = tf.keras.layers.Flatten()(input_layer)
first_dense = tf.keras.layers.Dense(128, activation=tf.nn.relu)(flatten_layer)
output_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(first_dense)
model = Model(inputs=input_layer, outputs=output_layer)

# prepare fashion mnist dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0

# configure, train, and evaluate the model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
