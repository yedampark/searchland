# Prediction
import numpy as np
import pandas as pd
import sys, requests, shutil, os, io
import urllib
import csv
import cv2
import matplotlib.pyplot as plt
import math
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import argparse
import random
import tensorflow as tf
from tensorflow.python.platform import app
import keras
import argparse
import time
from time import *

# the number of classes
n_class = 44

#os.system('cp ../scenes/*.jpg ../web/static')
os.system('mv ../scenes/*.csv ../csv') # move csv files

batch_size = 12

top_model_weights_path = '../web/bottleneck_fc_model.h5'
train_data_dir = '../data/train_images_model'
validation_data_dir = '../data/validation_images_model'
testfile = '../scenes'

subfile = '../data/prediction_new.csv'

def count(dir):
    i = 1
    count = []
    while i <= n_class:
        f = str(i)
        for root, dirs, files in os.walk(dir +'/'+ f):  # loop through startfolders
            for pic in files:
                count.append(pic)
        i += 1
    return len(count)

nb_train_samples = count(train_data_dir)
nb_validation_samples = count(validation_data_dir)


from keras import backend as K
def predict(image_path):
    print ('starting...')
    path, dirs, files = next(os.walk(image_path))
    file_len = len(files)
    print('Number of Testimages:', file_len)

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    generator = train_datagen.flow_from_directory(train_data_dir, batch_size=batch_size)
    label_map = (generator.class_indices)

    with open(subfile, 'w') as csvfile:
        newFileWriter = csv.writer(csvfile)
        newFileWriter.writerow(['id', 'landmarks', 'score'])
        file_counter = 0
        for root, dirs, files in os.walk(image_path):  # loop through startfolders
            for pic in files:
                t1 = clock()
                # loop folder and convert image
                path = image_path + '/' + pic
                orig = cv2.imread(path)
                image = load_img(path, target_size=(96, 96))
                image = img_to_array(image)

                # important! otherwise the predictions will be '0'
                image = image / 255
                image = np.expand_dims(image, axis=0)

                # classify landmark
                base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

                top_model = Sequential()
                top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
                top_model.add(Dense(256, activation='relu'))
                top_model.add(Dense(256, activation='relu'))
                top_model.add(Dense(n_class, activation='softmax'))

                model = Model(input=base_model.input, output=top_model(base_model.output))
                model.load_weights(top_model_weights_path)

                prediction = model.predict(image)

                class_predicted = prediction.argmax(axis=1)
                # class_predicted = np.argmax(prediction,axis=1)
                print (pic, class_predicted)

                inID = class_predicted[0]
                # print inID

                inv_map = {v: k for k, v in label_map.items()}
                # print class_dictionary

                label = inv_map[inID]

                score = max(prediction[0])
                if score >= 0.99:
                    scor = "{:.2f}".format(score)
                    out = str(label) + ' '+ scor
                    print (score)
                    newFileWriter.writerow([os.path.splitext(pic)[0], label, score])
                    print (os.path.splitext(pic)[0], out)
                    print('\n')
                    
                K.clear_session()

predict(testfile)
