# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 09:33:22 2022

@author: sutha
"""
#Importing the libraries
import tensorflow as tf



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
