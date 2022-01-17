# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 09:33:22 2022

@author: sutha
"""
#Importing the libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Building of CNN

#Initialising the CNN
classifier = Sequential()

#step 1 - Convolution - (check)
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

#Step 2 - pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 3 - flattening
classifier.add(Flatten())

#Step 4 - Full connection
classifier.add(Dense(units=148,activation='relu'))
classifier.add(Dense(units=5,activation='softmax'))

#Compiling the CNN
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('Garbage classification/train',target_size=(64, 64),batch_size=32,class_mode='categorical')

test_generator = test_datagen.flow_from_directory('Garbage classification/test',target_size=(64, 64),batch_size=32,class_mode='categorical')

classifier.fit(train_generator,steps_per_epoch=2023,epochs=30,validation_data=test_generator,validation_steps=504)

