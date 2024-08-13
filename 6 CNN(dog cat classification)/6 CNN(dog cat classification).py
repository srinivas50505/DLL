import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D,Flatten, Dense,MaxPooling2D,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import numpy as np
img_width,img_height=150,150
epochs=30
batch_size=32
model=Sequential()
model.add(Conv2D(32,(3,3),activation="relu",input_shape=(img_width,img_height,3)))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(256,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow_from_directory(r"C:\Users\satyakrishna\Desktop\Gokul\archive1\training_set\training_Set",
                                                  target_size=(img_width,img_height),
                                                  batch_size=batch_size,class_mode="binary")
validation_generator=train_datagen.flow_from_directory(r"C:\Users\satyakrishna\Desktop\Gokul\archive1\test_set\test_set",
                                                       target_size=(img_width,img_height),batch_size=batch_size,class_mode="binary")
model.fit(train_generator,steps_per_epoch=train_generator.samples// batch_size,epochs=epochs,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples// batch_size)
img=load_img("cat.jpeg",target_size=(img_width,img_height))
img=img_to_array(img)
img=np.expand_dims(img,axis=0)
img=img/255
prediction=model.predict(img)
if prediction[0][0]<0.5:
    print("Dog")
else:
    print("Cat")
