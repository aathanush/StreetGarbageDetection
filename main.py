# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 09:33:22 2022

@author: sutha
"""
#Importing the libraries
import tensorflow as tf
import numpy as np


# Building of CNN

#Initialising the CNN
#step 1 - Convolution - (check)
#Step 2 - pooling
#Step 3 - flattening
#Step 4 - Full connection
model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(32, (3,3),padding='same', activation='relu', input_shape=(128,128,3)),tf.keras.layers.MaxPooling2D((3,3)),tf.keras.layers.Conv2D(64, (3,3),padding='same', activation='relu', input_shape=(128,128,3)),tf.keras.layers.MaxPooling2D((3,3)), tf.keras.layers.Conv2D(128, (3,3),padding='same', activation='relu', input_shape=(128,128,3)),tf.keras.layers.MaxPooling2D((3,3)), tf.keras.layers.Conv2D(256, (3,3),padding='same', activation='relu', input_shape=(128,128,3)),tf.keras.layers.MaxPooling2D((3,3)), tf.keras.layers.Flatten(),tf.keras.layers.Dense(512, activation='relu'),tf.keras.layers.Dense(6, activation='softmax')])
#Compiling the CNN
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'],run_eagerly=True)


#Creating the train and test data
train=tf.keras.utils.image_dataset_from_directory( '/content/drive/MyDrive/Garbage classification/train',labels="inferred",label_mode="categorical",class_names=["cardboard","glass","metal","paper","plastic","trash"],color_mode="rgb",batch_size=32,image_size=(128, 128),)
test=tf.keras.utils.image_dataset_from_directory( '/content/drive/MyDrive/Garbage classification/test',labels="inferred",label_mode="categorical",class_names=["cardboard","glass","metal","paper","plastic","trash"],color_mode="rgb",batch_size=32,image_size=(128, 128),)

#Fitting the CNN to the images
model.fit(train,epochs=25,validation_data=test,batch_size=50)

print(model.summary())

#Prediction
classes=["cardboard","glass","metal","paper","plastic","trash"]
image_path=input("Enter the path of the image that you wish to classify")
image = tf.keras.preprocessing.image.load_img(image_path,target_size=(128,128))
img = tf.keras.preprocessing.image.img_to_array(image)
img = np.expand_dims(img, axis=0)
image = np.vstack([img])

t=model.predict(image)
t=list(t[0])
print("The predctions in decreasing order of probability are:")
a=classes.pop(t.index(max(t)))
t.pop(t.index(max(t)))
b=classes.pop(t.index(max(t)))
t.pop(t.index(max(t)))
c=classes.pop(t.index(max(t)))
t.pop(t.index(max(t)))
d=classes.pop(t.index(max(t)))
t.pop(t.index(max(t)))
print(f"{a}\n{b}\n{c}\n{d}\n")
