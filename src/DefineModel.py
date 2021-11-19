"""
Define a new model by extending class Model
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class MyModel(Model):

    def __init__(self, units, **kwargs):
        '''initializes the instance attributes'''
        super().__init__(**kwargs)
        self.my_input = Dense(units[0], activation='relu')

        self.hidden = []
        for idx in range(1, len(units) - 1):
            self.hidden.append(Dense(units[idx], activation='relu'))

        self.my_output = Dense(units[-1], activation='softmax')

    def call(self, inputs):
        '''defines the network architecture'''
        x = inputs
        for l in self.hidden:
            x = l(x)
        out = self.my_output(x)
        return out


# load data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0

# create model
model = MyModel([784, 32, 32, 10])

# train model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images.reshape(-1, 784), training_labels, epochs=5)
