import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)
model=Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(64,activation="relu"),
    Dropout(0.5),
    Dense(64,activation="relu"),
    Dropout(0.5),
    Dense(1)
])
model.compile(optimizer="adam",loss="mean_squared_error")
history=model.fit(x_train,y_train,epochs=100,batch_size=8,validation_split=0.2)
loss=model.evaluate(x_test,y_test)
print("Test Loss: ",loss)
