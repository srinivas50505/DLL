import numpy as np

import pandas as pd 

from numpy import unique, argmax

from tensorflow.keras.datasets.mnist import load_data 

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import Dense 

from tensorflow.keras.layers import Flatten 

from tensorflow.keras.layers import Dropout 

from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPool2D


from tensorflow.keras.datasets import mnist

import warnings
warnings.filterwarnings("ignore")


#loading the MNIST Dataset 

(train_x, train_y), (test_x, test_y) = mnist.load_data()
#printing the shapes 

print(train_x.shape, train_y.shape)

print(test_x.shape , test_y.shape)
#normalizing the pixel values of images 

train_x = np.expand_dims(train_x, axis=-1)
test_x = np.expand_dims(test_x, axis=-1)

#CNN Model 

model = Sequential()

shape = train_x.shape[1:]


#adding convolutional layer 

model.add(Conv2D(32, (3,3), activation='relu', input_shape= shape))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(48, (3,3), activation='relu'))

model.add(MaxPool2D((2,2)))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(500, activation='relu'))

model.add(Dense(10, activation='softmax'))
#compiling model 

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy',metrics= ['accuracy'] )

x=model.fit(train_x, train_y, epochs=10, batch_size = 128, verbose= 2 , validation_split = 0.1)
loss, accuracy= model.evaluate(test_x, test_y, verbose = 0)

print(f'Accuracy: {accuracy*100}')
