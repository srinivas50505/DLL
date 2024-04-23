#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# Load the pre-trained VGG16 model trained on ImageNet data
model = VGG16(weights='imagenet', include_top=True)

# Load and preprocess the image
img_path = 'C:\\Users\\acer\\Pictures\\Screenshots\\abc2.jpeg'  # Replace 'path_to_your_image.jpg' with the path to your image file
img = image.load_img(img_path, target_size=(224, 224))  # VGG16 expects input images to be of size 224x224
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make predictions
predictions = model.predict(x)

# Decode the predictions
decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get top 3 predictions
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")


# In[ ]:




