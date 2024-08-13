import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
objects=mnist
(train_img,train_lab),(test_img,test_lab)=objects.load_data()
print("traing_img",train_img.shape)
print("test_img",test_img.shape)
train_img=train_img/255
test_img=test_img/255
from keras.models import Sequential
from keras.layers import Flatten,Dense
model=Sequential()
input_layer=Flatten(input_shape=(28,28))
model.add(input_layer)
hidden_layer1=Dense(512,activation="relu")
model.add(hidden_layer1)
hidden_layer2=Dense(512,activation="relu")
model.add(hidden_layer2)
output_layer=Dense(512,activation="softmax")
model.add(output_layer)
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy"
             ,metrics=["accuracy"])
model.fit(train_img,train_lab,epochs=100)
loss_and_acc=model.evaluate(test_img,test_lab,verbose=2)
print("Test Loss",loss_and_acc[0])
print("Test Accuracy",loss_and_acc[1])
