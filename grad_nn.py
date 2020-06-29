import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_grad_nn(input):
    inputs = keras.Input(shape=(input.shape[1],))

    dense = layers.Dense(40, activation=None)
    trunk_1= dense(inputs)
    trunk_1=layers.BatchNormalization()(trunk_1)
    trunk_1=keras.activations.relu(trunk_1)
    trunk_2 = layers.Dense(40, activation=None)(trunk_1)
    trunk_2=layers.BatchNormalization()(trunk_2)
    trunk_2=keras.activations.relu(trunk_2)

    target_1 = layers.Dense(40, activation=None)(trunk_2)
    target_1=layers.BatchNormalization()(target_1)
    target_1=keras.activations.relu(target_1)
    target_2 = layers.Dense(40, activation=None)(target_1)
    target_2=layers.BatchNormalization()(target_2)
    target_2=keras.activations.relu(target_2)
    output_1 = layers.Dense(10, name="target_output")(target_2)

    attribute_1 = layers.Dense(40, activation=None)(trunk_2)
    attribute_1=layers.BatchNormalization()(target_1)
    attribute_1=keras.activations.relu(attribute_1)
    attribute_2 = layers.Dense(40, activation=None)(attribute_1)
    attribute_2=layers.BatchNormalization()(attribute_2)
    attribute_2=keras.activations.relu(attribute_2)
    output_2 = layers.Dense(10, name='attribute_output')(attribute_2)


    model = keras.Model(inputs=inputs, outputs=[output_1, output_2], name="german_credit")
    model.summary()

    losses = {
        "target_output": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "attribute_output": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    }
    lossWeights = {"target_output": 1.0, "attribute_output": 1.0}

    model.compile(
        loss=losses,
        loss_weights=lossWeights,
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    model.summary()

    return keras.Model(inputs=inputs, outputs=[output_1, output_2], name="german_credit")