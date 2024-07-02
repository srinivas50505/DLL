from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array 
import numpy as np
import warnings
# Set the dimensions of the input image 
img_width, img_height = 150, 150
# Define the number of epochs and batch size for training 
epochs = 50 
batch_size = 32 
# Create a CNN model 
model = Sequential()
# Add convolutional layers with pooling and dropout 
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3))) 
model.add(MaxPooling2D((2, 2))) 
model.add(Dropout(0.25)) 
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2))) 
model.add(Dropout(0.25)) 
model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2))) 
model.add(Dropout(0.25)) 
model.add(Conv2D(256, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2))) 
model.add(Dropout(0.25))
# Flatten the feature maps 
model.add(Flatten()) 
# Add a fully connected layer with dropout 
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.5))
# Add the output layer with sigmoid activation function 
model.add(Dense(1, activation='sigmoid'))
# Compile the model with binary cross-entropy loss and Adam optimizer 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Define the data generators for training and validation 
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True) 
test_datagen = ImageDataGenerator(rescale=1./255) 
train_generator = train_datagen.flow_from_directory(r"D:\LAB_EXP\LAB_EXP\DLLAB\archive1\training_set\training_set", target_size=(img_width, img_height), 
batch_size=batch_size, class_mode='binary') 
validation_generator = test_datagen.flow_from_directory(r"D:\LAB_EXP\LAB_EXP\DLLAB\archive1\test_set\test_set", target_size=(img_width, 
img_height), batch_size=batch_size, class_mode='binary')
img = load_img(r"D:\LAB_EXP\LAB_EXP\DLLAB\archive1\test_set\test_set\cats\cat.4810.jpg", target_size=(img_width, img_height))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
img /= 255.
prediction = model.predict(img)
if prediction[0] < 0.5:
    print("The image is a cat.")
else:
    print("The image is a dog.")
