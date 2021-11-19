"""
Define a new layer by extending Layer class
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


# Define a new layer
class MySimpleLayer(Layer):

    def __init__(self):
        super(MySimpleLayer, self).__init__()

    def build(self, input_shape):  # Create the state of the layer (weights)
        pass

    def call(self, inputs):  # Defines the computation from inputs to outputs
        return tf.keras.backend.maximum(inputs, 0)  # this is RELU


# load data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0

# create model
input_layer = tf.keras.Input(shape=(28, 28))
x = tf.keras.layers.Flatten()(input_layer)
x = MySimpleLayer()(x)  # add a layer
out_layer = tf.keras.layers.Dense(10, activation='softmax')(x)
model = Model(inputs=input_layer, outputs=out_layer)

# train model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
tf.keras.layers.ReLU()
