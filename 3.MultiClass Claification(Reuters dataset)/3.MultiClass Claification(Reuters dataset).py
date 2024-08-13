import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Input
from tensorflow.keras.utils import to_categorical
(x_train,y_train),(x_test,y_test)=reuters.load_data(num_words=10000)
def Vectorize_Sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i ,sequence in enumerate(sequences):
        results[i,sequence]=1
    return results
x_train=Vectorize_Sequences(x_train)
x_test=Vectorize_Sequences(x_test)
num_classes=np.max(y_train)+1
y_train=to_categorical(y_train,num_classes)
y_test=to_categorical(y_test,num_classes)
model=Sequential([
                 Input(shape=(10000,)),
                 Dense(64,activation="relu"),
                 Dropout(0.5),
                 Dense(64,activation="relu"),
                 Dropout(0.5),
                 Dense(num_classes,activation="softmax")])
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
history=model.fit(x_train,y_train,epochs=10,batch_size=512,validation_split=0.2)
loss,accuracy=model.evaluate(x_test,y_test)
print("test Accuracy: ",accuracy*100)
