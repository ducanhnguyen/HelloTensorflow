"""
Define a model with multiple inputs
"""

from keras.layers import concatenate
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model


def initialize_base_network():
    input = Input(shape=(28, 28,))
    x = Flatten(name="flatten_input")(input)
    x = Dense(128, activation='relu')(x)
    return Model(inputs=input, outputs=x)


base_network = initialize_base_network()

# create the left input and point to the base network
input_a = Input(shape=(28, 28,), name="left_input")
vect_output_a = base_network(input_a)

# create the right input and point to the base network
input_b = Input(shape=(28, 28,), name="right_input")
vect_output_b = base_network(input_b)

output = concatenate(inputs=[vect_output_a, vect_output_b])
model = Model([input_a, input_b], output)

model.summary()
