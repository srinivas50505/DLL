#pip install tensorflow
#!pip install keras
#!pip install numpy

import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

# Load the IMDb dataset
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)

# Pad sequences to a fixed length
max_sequence_length = 500  # Adjust as needed
training_data = pad_sequences(training_data, maxlen=max_sequence_length)
testing_data = pad_sequences(testing_data, maxlen=max_sequence_length)

# Concatenate padded sequences
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

data = vectorize(data)
targets = np.array(targets).astype("float32")

# Split into training and testing sets
test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

# Define the model architecture
model = models.Sequential()
model.add(layers.Dense(50, activation="relu", input_shape=(10000, )))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model
results = model.fit(
    train_x, train_y,
    epochs=2,
    batch_size=500,
    validation_data=(test_x, test_y)
)

print("Test-Accuracy:", np.mean(results.history["val_accuracy"]))
