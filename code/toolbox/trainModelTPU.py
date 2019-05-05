'''
Import the class1 and class2 patterns (pkl file) and train the model on TPU.
(Please copy this file to Colab notebooks)
'''

import os
import pickle
import tensorflow as tf
import numpy as np
from numpy import newaxis

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

# Keras
from tensorflow.python import keras
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import models, layers, regularizers

# Connect to Google Drive
from google.colab import drive
drive.mount('/content/drive')
PATH_TO_GOOGLE_DRIVE = '/content/drive/My Drive/Colab Notebooks/'



'''
Search all .pkl files in current folder and merge them as one array of patterns and labels
'''
def search_merge_training_data():

    # Init
    t_Patterns = np.array([])
    t_Labels = []

    # Find all pkl file in the current folder
    for t_File in os.listdir(PATH_TO_GOOGLE_DRIVE):

        if t_File.endswith(".pkl"):

            with open(PATH_TO_GOOGLE_DRIVE + t_File, "rb") as t_PKLFile:

                t_Dict = pickle.load(t_PKLFile)
                t_Patterns = np.vstack([t_Patterns, t_Dict["Patterns"]]) if t_Patterns.size else t_Dict["Patterns"]
                t_Labels = np.concatenate((t_Labels, t_Dict["Labels"]), axis=None)

                print(t_File, "has been loaded!"," Patterns shape:",
                              t_Patterns.shape, " Labels len:", len(t_Labels))

    print("All training sets are combined!")

    return t_Patterns, t_Labels



'''
Remove mean and variance of data and save the Scalers.
'''
def normalize_data(a_Patterns, a_Labels):

    # Normalize patterns in each feature space
    t_Scalers = {}

    for i in range(a_Patterns.shape[2]):

        t_Scalers[i] = StandardScaler()
        a_Patterns[:, :, i] = t_Scalers[i].fit_transform(a_Patterns[:, :, i])

    print("Patterns normalization done!")

    # Save scaler for validation set
    t_ScalerFile = PATH_TO_GOOGLE_DRIVE + 'Scalers.sav'
    pickle.dump(t_Scalers, open(t_ScalerFile, 'wb'))

    print("Scalers saved!")

    # Normalize labels
    t_Encoder = LabelEncoder()
    a_Labels = t_Encoder.fit_transform(a_Labels)

    print("Labels normalization done!")

    return a_Patterns, a_Labels



'''
Search and combine all training set then normalize
'''
def preprocess_training_data():

    t_Patterns, t_Labels = search_merge_training_data()
    t_Patterns, t_Labels = normalize_data(t_Patterns, t_Labels)

    return t_Patterns, t_Labels



'''
Use preprocessed patterns and labels to train the CNN model
'''
def train_model(a_Patterns, a_Labels, a_KSplit):

    # Check if TPU available
    try:
      t_DeviceName = os.environ['COLAB_TPU_ADDR']
      t_TPUaddress = 'grpc://' + t_DeviceName
      print('Found TPU at: {}'.format(t_TPUaddress))
    except KeyError:
      print('TPU not found')

    # Reshape training data to 4D Depth, width, height, num of patterns
    a_Patterns = a_Patterns[:,:,:,newaxis]

    # Set up K Fold Validation
    t_KFold = StratifiedKFold(n_splits=a_KSplit, shuffle=True, random_state=np.random.seed(7))

    # Start training by K times and save the model with highest ACC
    t_BestACC = 0
    t_BestModel = None
    for trainIdx, testIdx in t_KFold.split(a_Patterns, a_Labels):

        # Init empty model
        t_Model = models.Sequential()

        # Define input layer
        t_Model.add(Conv2D(32, kernel_size=(3,3), activation='relu',
                                                input_shape=a_Patterns[0].shape))
        t_Model.add(ZeroPadding2D(2))

        # Define hidden layers
        t_Model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
        t_Model.add(ZeroPadding2D(2))

        t_Model.add(MaxPooling2D(pool_size=(3,3), strides=None, padding='valid'))
        t_Model.add(Dropout(0.5, noise_shape=None, seed=None))

        t_Model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        t_Model.add(ZeroPadding2D(2))

        t_Model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        t_Model.add(ZeroPadding2D(2))

        t_Model.add(MaxPooling2D(pool_size=(3,3), strides=None, padding='valid'))
        t_Model.add(Dropout(0.25, noise_shape=None, seed=None))

        t_Model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
        t_Model.add(ZeroPadding2D(2))

        t_Model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
        t_Model.add(ZeroPadding2D(2))

        t_Model.add(GlobalMaxPooling2D())

        t_Model.add(Flatten(data_format=None))

        t_Model.add(Dense(32))
        t_Model.add(Dropout(0.5, noise_shape=None, seed=None))

        # Define output layer
        t_Model.add(Dense(2, activation='softmax'))

        # Set learning rate
        opt = tf.train.AdamOptimizer(0.001)

        t_Model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

        # Convert to TPU model
        t_Resolver = tf.contrib.cluster_resolver.TPUClusterResolver(t_TPUaddress)
        t_Strategy = tf.contrib.tpu.TPUDistributionStrategy(t_Resolver)
        tpu_model = tf.contrib.tpu.keras_to_tpu_model(t_Model, strategy=t_Strategy)

        # Print model structure
        print(t_Model.summary())

        tpu_model.fit(a_Patterns[trainIdx], a_Labels[trainIdx], epochs=25, batch_size=1024,
                      validation_data=(a_Patterns[testIdx], a_Labels[testIdx]))
        scores = tpu_model.evaluate(a_Patterns[testIdx], a_Labels[testIdx], verbose=0)

        # Update best model
        print("%s: %.2f%%" % (tpu_model.metrics_names[1], scores[1]*100))
        if scores[1] > t_BestACC:
            t_BestACC = scores[1]
            t_BestModel = tpu_model

    print('BestACC =', t_BestACC)

    # Save model with the highest accuracy to google drive
    t_BestModel.save(PATH_TO_GOOGLE_DRIVE + 'cnnModel.h5')







X, y = preprocess_training_data()

t_KFold = 14
train_model(X, y, t_KFold)
