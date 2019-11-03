import numpy as np
import keras
from glob import glob
import random
import os
import cv2
import pandas as pd
from keras.utils.np_utils import to_categorical
import imgaug.augmenters as iaa
from enet_preprocess import *

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path, batch_size = 4, dataframe = None, shuffle=True, dim = (256,256), split = False):
        'Initialization'
        self.path = path
        self.batch_size = batch_size
        self.dataframe = dataframe
        self.list_IDs = self.dataframe['id_code']
        self.shuffle = shuffle
        self.dim = dim
        self.on_epoch_end()
        self.split = split
        self.img_width_split = self.dim[0]//2


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs.iloc[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        #print(label_df.shape)
        
        X = []
        y = []
        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            img = cv2.cvtColor(cv2.imread(self.path + '/{}.png'.format(ID)), cv2.COLOR_BGR2RGB)

            img = preprocess_image(img, resize = self.dim)
            X.append(img)
            # Store class              
            y.append(np.eye(5)[int(self.dataframe.loc[self.dataframe['id_code'] == ID]['diagnosis'])])
            # except Exception as e:
            #     print(e)
        #print(X, y)
        X, y = np.array(X), np.array(y)

        if self.split == True:
            return([X[:, 0:self.img_width_split, 0:self.img_width_split, :], X[:, self.img_width_split:self.img_width_split*2, 0:self.img_width_split, :], 
                X[:, 0:self.img_width_split, self.img_width_split:self.img_width_split*2, :], X[:, self.img_width_split:self.img_width_split*2, self.img_width_split:self.img_width_split*2, :]], y)
        else:
            return np.array(X), np.array(y)

    
