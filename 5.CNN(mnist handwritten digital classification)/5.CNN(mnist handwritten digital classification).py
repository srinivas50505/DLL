import tensorflow as tf 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D 
# Load the MNIST dataset and split it into train and test sets 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# Reshape the input data to be 4-dimensional (batch_size, height, width, channels) 
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) 
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) 
# Normalize the pixel values to be between 0 and 1 
x_train = x_train / 255.0 
x_test = x_test / 255.0 
# Define the CNN architecture 
model = Sequential([Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), 
MaxPooling2D((2, 2)),Conv2D(64, (3, 3), activation='relu'),MaxPooling2D((2, 2)),Flatten(), 
Dense(128, activation='relu'),Dense(10, activation='softmax') ]) 
# Compile the model 
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])   
# Train the model on the training data 
model.fit(x_train, y_train, epochs=5) 
# Evaluate the model on the test data 
test_loss, test_acc = model.evaluate(x_test, y_test) 
print('Test accuracy:', test_acc) 
# Predict new data 
import numpy as np 
from PIL import Image 
# Load a sample image and preprocess it 
import matplotlib.pyplot as plt 
predictions = model.predict(x_test) 
predicted_classes = np.argmax(predictions, axis=1) 
print('Predicted classes:', predicted_classes) 
plt.matshow(x_test[0])
