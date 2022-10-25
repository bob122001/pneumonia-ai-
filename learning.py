from wsgiref import validate
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout
import tensorflow as tf
import pickle
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator 

dataset_path = '/home/robert/Pneumonia AI/chest_xray'
train_path = '/home/robert/Pneumonia AI/chest_xray/train'
val_path = '/home/robert/Pneumonia AI/chest_xray/val'
test_path = '/home/robert/Pneumonia AI/chest_xray/test'

batch = 100

num_classes = 2

epochs = 10

img_width, img_height = 128, 128

normalization_layer = tf.keras.layers.Rescaling(1./255)

train = tf.keras.utils.image_dataset_from_directory(train_path, image_size=(img_height, img_width), label_mode='binary', labels='inferred')

validate = tf.keras.utils.image_dataset_from_directory(test_path, image_size=(img_height, img_width), label_mode='binary', labels='inferred')

test = tf.keras.utils.image_dataset_from_directory(val_path, image_size=(img_height, img_width), label_mode='binary', labels='inferred')

class_weights = {0: (3875/1341), 1: (1341/3875)}

model = Sequential([
    normalization_layer,
    Conv2D(16, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.5),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

model.fit(train, batch_size=batch, validation_data=validate, epochs=epochs, verbose=1, class_weight=class_weights)

model.evaluate(test)

pickle.dump(model, open('model.sav', 'wb'))



