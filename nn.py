import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_nn(input):
    inputs = keras.Input(shape=(input.shape[1],))

    dense = layers.Dense(40, activation=None)
    a = dense(inputs)
    a=layers.BatchNormalization()(a)
    a=keras.activations.relu(a)

    b = layers.Dense(40, activation=None)(a)
    b=layers.BatchNormalization()(b)
    b=keras.activations.relu(b)

    c = layers.Dense(40, activation=None)(b)
    c=layers.BatchNormalization()(c)
    c=keras.activations.relu(c)

    d = layers.Dense(40, activation=None)(c)
    d=layers.BatchNormalization()(d)
    d=keras.activations.relu(d)

    outputs = layers.Dense(10)(d)
    model = keras.Model(inputs=inputs, outputs=outputs, name="german_credit")
    
    model.summary()
    
    return keras.Model(inputs=inputs, outputs=outputs, name="german_credit_nn") 