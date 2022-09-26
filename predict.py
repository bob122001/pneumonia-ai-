import pickle
from keras.preprocessing import image
import numpy as np 
from keras.applications.resnet import decode_predictions, preprocess_input
from keras.utils import load_img, img_to_array 


img_size = (128, 128)
model = pickle.load(open('model.sav', 'rb'))


test_image_path = '/home/robert/Pneumonia AI/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg'
test_img = load_img(test_image_path, target_size=(img_size))

test_img_array = img_to_array(test_img)
test_img_ex = np.expand_dims(test_img_array, axis=0)

test_img_pre = preprocess_input(test_img_ex)


def predict_image(image_path):
    prediction = model.predict(test_img_pre)
    print(decode_predictions(prediction, top=1))
    #print(prediction.shape)
    

predict_image(test_img)