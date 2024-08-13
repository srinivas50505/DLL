import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from tensorflow.keras.preprocessing import sequence

max_features=20000
max_len=100
batch_size=32
epochs=5

print("Loading data")
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
print("Pad Sequences (Sample X time)")
x_train=sequence.pad_sequences(x_train,maxlen=max_len)
x_test=sequence.pad_sequences(x_test,maxlen=max_len)
print("Build Model.......")
model=Sequential()
model.add(Embedding(max_features,128))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) 
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("Test score: ",score)
print("Accuracy: ",acc)
