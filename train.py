import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from nn import create_nn
from grad_nn import create_grad_nn

GERMAN_CREDIT_DATA_WITH_RISK_PATH='Data/german_credit_data_with_risk.csv'

def read_data(data_path):
    df=pd.read_csv(data_path, index_col=0)
    return df

def process_german_credit_data(df):
    LE = LabelEncoder()

    df["Risk"] = LE.fit_transform(df["Risk"])
    df["Housing"] = LE.fit_transform(df["Housing"])

    df["Checking account"] = df["Checking account"].map({ 'little' : 1, 'moderate': 2 , 'rich': 3})

    df["Checking account"]  = df["Checking account"].fillna(0)

    df["Purpose"] = df["Purpose"].map({ 'radio/TV' : 1, 'education' : 2 , 'furniture/equipment' : 1 , 'car' : 0 , 'business' :3,
        'domestic appliances' : 1 , 'repairs' : 1 , 'vacation/others' : 4})

    df["Saving accounts"] = df["Saving accounts"].map({ 'little' : 1, 'moderate': 2 , 'rich': 4 , 'quite rich' : 3})

    df["Saving accounts"] = df["Saving accounts"].fillna(0)

    df["Sex"] = df["Sex"].map({ 'male' : 0, 'female': 1 })

    print(df.head())

    print(df.shape)
    X = df.drop("Risk", axis = 1)
    X=X.values
    print(X.shape)

    y = df[["Risk"]]
    y=y.values
    print(y.shape)

    
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X,y, random_state=42)

data_df = read_data(GERMAN_CREDIT_DATA_WITH_RISK_PATH)
X_train, X_test, y_train, y_test = process_german_credit_data(data_df)

model_nn = create_nn(X_train)

model_nn.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)
history_nn = model_nn.fit(X_train, y_train, batch_size=64, epochs=50, validation_split=0.2)
test_scores_nn = model_nn.evaluate(X_test, y_test, verbose=2)

model_grad_nn = create_grad_nn(X_train)

losses = {
	"target_output": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	"attribute_output": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
}
lossWeights = {"target_output": 1.0, "attribute_output": 1.0}

model_grad_nn.compile(
    loss=losses,
    loss_weights=lossWeights,
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

history_grad_nn = model_grad_nn.fit(X_train, y_train, batch_size=64, epochs=50, validation_split=0.2)
test_scores_grad_nn = model_grad_nn.evaluate(X_test, y_test, verbose=2)


print("Test loss NN:", test_scores_nn[0])
print("Test accuracy NN:", test_scores_nn[1])

print("Test loss Grad NN:", test_scores_grad_nn[0])
print("Test accuracy Grad NN:", test_scores_grad_nn[1])





