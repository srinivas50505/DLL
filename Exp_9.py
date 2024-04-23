#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set parameters for the model and training
max_features = 10000  # Number of words to consider as features
maxlen = 100  # Cut texts after this number of words (among top max_features most common words)
embedding_dim = 50  # Dimensionality of the word embeddings
batch_size = 32
epochs = 5

# Load the IMDb dataset
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to have consistent length for the input to the Embedding layer
print('Pad sequences (samples x time)')
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Build the model
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, embedding_dim))  # Embedding layer
model.add(Flatten())  # Flatten the 3D embedding tensor to 2D
model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

# Evaluate the model
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


# In[ ]:




