import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input,decode_predictions

model=VGG16(weights="imagenet",include_top=True)
img_path="parrot.jpeg"
img=image.load_img(img_path,target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)
predictions=model.predict(x)
decode_predictions=decode_predictions(predictions,top=3)[0]
for i ,(imagenet_id,label,score) in enumerate (decode_predictions):
    print(f"{i+1} : {label} ({score : .2f})")
