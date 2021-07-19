# Model Training - Transfer Learning
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import csv
import os
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
import math
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import argparse
import random
import tensorflow as tf
import keras
import shutil

# the number of classes
n_class = 44

train_data_dir = '../data/train_images_model'
validation_data_dir = '../data/validation_images_model'

# counts the number of files in subdirectories of a given directory.
# outputs the count of images and label of each image in order of reading.
def count(dir):
    i = 1
    count = []
    while i <= n_class:
        f = str(i)
        #print (f)
        for root, dirs, files in os.walk(dir +'/'+ f):  # loop through startfolders
            for pic in files:
                count.append(f)
            i += 1
    print (len(count))
    return ([len(count),count])

nb_train_samples = count(train_data_dir)
nb_validation_samples = count(validation_data_dir)

# nb_train/validation_samples is a list of lists of the format [len(count),count]. len(count) is the total number of images in train/validation folder.
# count is a list of class labels i.e. landmark IDs of the images read.

# In the next steps, a batch size will be defined, which needs to be a factor of both the number of train and validation images.

# Converting images to vectors using weights from ImageNet on VGG16
img_width, img_height = 96, 96 # dimensions of downloaded images.
top_model_weights_path = 'bottleneck_fc_model.h5' # A file with this name would be saved later in the code
epochs = 5
batch_size = 990 # As found in the previous step
def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1. / 255,
                                 rotation_range=30,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 zoom_range = 0.5,
                                 brightness_range = [0.5,1.5])

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(96,96,3))
    print ('start1')
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,    # this means our generator will only yield batches of data, no labels
        shuffle=False)      # our data will be in order

    # the predict_generator method returns the output of a model, given a generator that yields batches of numpy data
    print ('start2')
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples[0] // batch_size+1)
    print ('bottleneck_features_trained')

    with open('bottleneck_features_train.npy', 'wb') as features_train_file:
        np.save(features_train_file, bottleneck_features_train)
    print ('Train done')

    datagen = ImageDataGenerator(rescale=1. / 255) #No image augmentation in validation dataset
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print ('validation predict start')
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples[0] // batch_size+1)

    with open('bottleneck_features_validation.npy', 'wb') as features_validation_file:
        np.save(features_validation_file, bottleneck_features_validation)
    print ('validation done')

save_bottleneck_features()


# Initializing the weights on top 3 layers
epochs = 5
batch_size = 990

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.array(nb_train_samples[1])
    train_labels = [str(int(train_label) - 1) for train_label in train_labels]
    # Had to subtract 1 because class labels should start from 0. In this case, class labels had a range from 1 to n_class.
    # print (train_labels)
    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array(nb_validation_samples[1])
    validation_labels = [str(int(validation_label) - 1) for validation_label in validation_labels]

    # print (validation_labels)
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(n_class, activation='softmax')) # n_class is the number of classes fed to the model
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    train_labels = to_categorical(train_labels, n_class)
    validation_labels = to_categorical(validation_labels, n_class)

    print ('model fit starting')
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

train_top_model()


# Compile and train the entire model
img_width, img_height = 96, 96
top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = '../data/train_images_model'
validation_data_dir = '../data/validation_images_model'
batch_size = 240
epochs = 32

def trainCNN():
    # build the VGG16 network
    base_model = applications.VGG16(weights='imagenet',include_top= False,input_shape=(96,96,3))

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dense(n_class, activation='softmax')) # n_class is the number of classes fed to the model
    top_model.load_weights(top_model_weights_path) # Load the weights initialized in previous steps

    model = Model(input= base_model.input, output= top_model(base_model.output))

    # set the first 16 layers to non-trainable (weights will not be updated) -> 1 conv layer and three dense layers will be trained
    for layer in model.layers[:16]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer and a very slow learning rate.
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.001, beta_1=0.9,beta_2=0.999,epsilon=1e-8, decay=0.0),
                  metrics=['accuracy'])
    print ('Compilation done.')

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=90,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        zoom_range = 0.5)

    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    np.save('class_indices.npy', train_generator.class_indices)

    validation_generator = valid_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    print ('Model fit begins...')
    model.fit_generator(
        train_generator,
        steps_per_epoch=340,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=150,
        callbacks=[ModelCheckpoint(filepath=top_model_weights_path, save_best_only=True, save_weights_only=True)]
        )

    model.save_weights(top_model_weights_path)

trainCNN()

print('train.py end')
