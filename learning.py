import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import tensorflow as tf
import pickle

dataset_path = '/home/robert/Pneumonia AI/chest_xray'
train_path = '/home/robert/Pneumonia AI/chest_xray/train'
val_path = '/home/robert/Pneumonia AI/chest_xray/val'
test_path = '/home/robert/Pneumonia AI/chest_xray/test'

batch = 100

num_classes = 2

epochs = 2

img_width, img_height = 128, 128

normalization_layer = tf.keras.layers.Rescaling(1./255)


CATEGORIES = ['NORMAL', 'PNEUMONIA']

train = tf.keras.utils.image_dataset_from_directory(train_path, image_size=(img_height, img_width))

validate = tf.keras.utils.image_dataset_from_directory(test_path, image_size=(img_height, img_width))


model = Sequential([
    normalization_layer,
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(),
    Dropout(0.5),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes)
])





model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

model.fit(train, batch_size=batch, epochs=epochs, verbose=1, validation_data=validate)

pickle.dump(model, open('model.sav', 'wb'))



