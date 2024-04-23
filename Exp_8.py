#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example text data
text_data = ["The cat sat on the mat.", "The dog ate my homework."]

# Word-level one-hot encoding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
word_sequences = tokenizer.texts_to_sequences(text_data)
word_one_hot = tokenizer.sequences_to_matrix(word_sequences, mode='binary')

print("Word-level one-hot encoding:")
print(word_one_hot)

# Character-level one-hot encoding
max_length = 50  # Maximum number of characters in a sample
max_chars = 128  # Maximum number of ASCII characters

char_one_hot = np.zeros((len(text_data), max_length, max_chars), dtype=np.float32)
for i, sample in enumerate(text_data):
    for j, char in enumerate(sample):
        if j < max_length:
            index = ord(char)
            if index < max_chars:
                char_one_hot[i, j, index] = 1.0

print("\nCharacter-level one-hot encoding:")
print(char_one_hot)


# In[ ]:




