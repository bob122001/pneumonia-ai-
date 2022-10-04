import pickle
from keras.preprocessing import image
import numpy as np 
from keras.applications.resnet import decode_predictions, preprocess_input
from keras.utils import load_img, img_to_array 
import tensorflow as tf


img_size = (256, 256)
model = pickle.load(open('model.sav', 'rb'))

threshold = 0.5

def predict_image(image_path):
    img = load_img(image_path, target_size=(img_size))

    img_array = img_to_array(img)
    img_ex = np.expand_dims(img_array, axis=0)

    img_pre = preprocess_input(img_ex)
   
    prediction = model.predict(img_pre)

    print(prediction[0])
    print(image_path)
    
    if prediction[0] > threshold:
        return("Positive")
    else:
        return("Negative")
    