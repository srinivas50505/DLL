import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,Flatten
num_words=1000
max_length=200
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=1000)
x_train=tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_length)
x_test=tf.keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_length)
model =Sequential()
model.add(Embedding(input_dim=num_words,output_dim=32))
model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=10,batch_size=128,validation_data=(x_test,y_test))
loss,accuracy=model.evaluate(x_test,y_test)
print("Test Accuracy : ",accuracy*100)
