import numpy as np
import keras
from glob import glob
import random
import os
import cv2
import pandas as pd
from keras.utils.np_utils import to_categorical

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path, batch_size = 16, dim = (128, 128), channels = 4):
        'Initialization'
        self.path = path
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = channels


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(glob(self.path + '*.jpeg')) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        # Generate data
        #print(indexes_i, indexes_j)

        # Generate data
        X, y = self.__data_generation()

        return X, y

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization
        path = glob(self.path + '*.jpeg')

        label_df = pd.read_csv('/media/parth/DATA/DiabeticR/trainLabels.csv')
        
        X = []
        y = []
        # Generate data
        while(len(X)) != self.batch_size:
            index = random.randint(0, len(path))
            #print(os.path.basename(path[index]).split('.')[0])
            # Store sample
            try:
                X.append(cv2.imread(path[index]))
                # Store class              
                y.append(to_categorical(label_df.iloc[label_df.index[label_df['image'] == os.path.basename(path[index]).split('.')[0]], 1].item(), num_classes=5))
            except Exception as e:
                print(e)

        return np.array(X), np.array(y)


if __name__ == '__main__':

    D = DataGenerator('/media/parth/DATA/DiabeticR/train/')

    