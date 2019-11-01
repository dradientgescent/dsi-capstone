import numpy as np
import pandas as pd
import keras 
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
from keras import layers
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, ELU, Softmax
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import keras
import tensorflow as tf
import cv2
from tqdm import tqdm
from enet_preprocess import *
from keras_radam import RAdam
from enet import create_model
from group_norm import GroupNormalization
from enet_preprocess import *
sys.path.append(os.path.abspath('../../efficientnet/'))
import efficientnet.keras as efn 
from efficientnet.model import EfficientNetB5
from dataloader import DataGenerator
from sklearn.metrics import confusion_matrix
from losses import *
sys.path.append(os.path.abspath('../../BioExp/'))
from BioExp.spatial.cam import *


BATCH_SIZE = 1
IMG_WIDTH, IMG_HEIGHT = 460, 460
TRAIN_IMG_PATH = '/media/brats/mirlproject2/aptos_2019/train_cropped'
CHANNELS = 3

model = create_model(dim = (460, 460), weights = np.ones(5))
model.load_weights('/home/brats/parth/dsi-capstone/saved_models/effnet_weights_functional.h5')
f_loss = categorical_focal_loss()
model.compile(loss=f_loss,
	              optimizer=RAdam(lr=0.00005), 
	              metrics=[f_loss, 'acc'])
#print(model.summary())

label_df = pd.read_csv('/media/brats/mirlproject2/aptos_2019/train.csv')

train, test = train_test_split(label_df, test_size=0.15, random_state=42)

test_index = 8

# val_datagen = DataGenerator(TRAIN_IMG_PATH, batch_size = 1, dataframe = test, dim = (IMG_HEIGHT, IMG_WIDTH), shuffle= False)
# print('Found {} Val Images\n'.format(val_datagen.__len__()*BATCH_SIZE))

# predict_array = model.predict_generator(val_datagen)

for test_index in range(len(test)):


	img = cv2.cvtColor(cv2.imread(TRAIN_IMG_PATH + '/{}.png'.format(test['id_code'].iloc[test_index])), cv2.COLOR_BGR2RGB)

	img = preprocess_image(img, resize = (460, 460))[None, ...]

	print('Test image {}: Original Label = {}, Predicted Label = {}'.format(test_index, test['diagnosis'].iloc[test_index], 
		np.argmax(model.predict(img))))


	# y_true = np.array(test['diagnosis'])	
	# print(y_true.shape)

	# # BATCH_SIZE = 1
	# # IMG_WIDTH, IMG_HEIGHT = 460, 460
	# # TRAIN_IMG_PATH = '/media/brats/mirlproject2/aptos_2019/train_cropped'

	# print('---------------------Initialized Testing---------------------\n')
	# # Add Image augmentation to our generator
	# train_datagen = DataGenerator(TRAIN_IMG_PATH, batch_size = 1, dataframe = train, dim = (IMG_HEIGHT, IMG_WIDTH))
	# print('Found {} Train Images'.format(train_datagen.__len__()*BATCH_SIZE))
	# val_datagen = DataGenerator(TRAIN_IMG_PATH, batch_size = 1, dataframe = test, dim = (IMG_HEIGHT, IMG_WIDTH), shuffle= False)
	# print('Found {} Val Images\n'.format(val_datagen.__len__()*BATCH_SIZE))

	# y_pred = np.squeeze(np.argmax(model.predict_generator(val_datagen), axis = 1))

	# print(sum((y_pred==y_true)*1)/len(y_true))
	# # print(model.evaluate_generator(val_datagen))
	# print(y_pred.shape)


	# print(confusion_matrix(y_true, y_pred))


	get_cam(model, img, layer_idx=-1, custom_objects = {'RAdam':RAdam, 'categorical_focal_loss_fixed': categorical_focal_loss(alpha=np.ones(5))}, 
		label = '{}_{}_{}'.format(test_index, test['diagnosis'].iloc[test_index], np.argmax(model.predict(img))))