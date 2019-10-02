from keras_applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121
from keras_applications.resnet import ResNet101
from keras.callbacks import  ModelCheckpoint
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras.models import model_from_json,load_model
from dataloader import DataGenerator
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from keras import backend as K
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss


def create_model(class_weights):

	# create the base pre-trained model
	base_model = ResNet101(weights=None, include_top=False, backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)

	# add a global spatial average pooling layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	# let's add a fully-connected layer
	# x = Dropout(0)(x)
	x = Dense(1024, activation='relu')(x)
	#x = Dropout(0.15)(x)
	x = Dense(256, activation='relu')(x)
	# and a logistic layer -- let's say we have 200 classes
	predictions = Dense(5, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=predictions)

	sgd = SGD(lr=0.01, momentum=0.9, decay=5e-6, nesterov=False)
	adam = Adam(lr=1e-3)
	model.compile(optimizer=adam, loss=weighted_categorical_crossentropy(class_weights), metrics = ['accuracy'])

	return model


class Training(object):
    

    def __init__(self, model, nb_epoch, batch_size, savepath, load_model_resume_training=None, weight_path=None):

        
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.savepath = savepath

        #loading model from path to resume previous training without recompiling the whole model
        if load_model_resume_training is not None:
            self.model = model        	
            self.model.load_weights(weight_path)
            print("pre-trained model loaded!")
        else:
            self.model = model
            #self.model.load_weights('/home/parth/Interpretable_ML/Brain-tumor-segmentation/checkpoints/Unet_cc/SimUnet.01_0.095.hdf5')
            print("Model compiled!")

    def fit(self, train_gen, val_gen):
		
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

        train_generator = train_gen
        val_generator = val_gen
        checkpointer = ModelCheckpoint(filepath=self.savepath, verbose=1, save_best_only = True)
        self.model.fit_generator(train_generator,
                                 epochs=self.nb_epoch, validation_data=val_generator, 
                                 steps_per_epoch = len(train_generator), validation_steps = len(val_generator),  
                                 verbose=1, callbacks=[checkpointer, reduce_lr])



if __name__ == '__main__':

	label_df = pd.read_csv('/media/bmi/poseidon/DiabeticR/trainLabels.csv')

	label_df["image"] = label_df["image"].apply(lambda name : name + '.jpeg')

	label_df["level"] = label_df["level"].apply(lambda label : str(label))

	train, test = train_test_split(label_df, test_size=0.25, random_state=42)

	print(train.shape, test.shape)

	y_train = np.array(label_df['level'])

	sample_array = []

	# for i in range(5000):
	# 	try:
	# 		sample_array.append(cv2.resize(cv2.imread('/media/bmi/poseidon/DiabeticR/train_cropped/' 
	# 			+ train["image"].iloc[i]), (512,512)))
	# 	except Exception as e:
	# 		print(e)

	# sample_array = np.array(sample_array)

	# print(sample_array.shape)

	

	batch_size = 2

	
	#train_generator = DataGenerator('/media/parth/DATA/DiabeticR/train_resized/', batch_size = 16)
	#val_generator = DataGenerator('/media/parth/DATA/DiabeticR/val_resized/', batch_size = 16)

	train_gen = ImageDataGenerator(
		samplewise_center=True,
		samplewise_std_normalization=True,
		rotation_range=10,
		width_shift_range=10,
		height_shift_range=10,
		horizontal_flip=True
		)

	val_gen = ImageDataGenerator(
		samplewise_center=True,
		samplewise_std_normalization=True
		)

	# train_gen.fit(sample_array)
	# val_gen.fit(sample_array)

	train_generator = train_gen.flow_from_dataframe(
			dataframe=train,
		directory='/media/bmi/poseidon/DiabeticR/train_cropped',
		x_col="image",
		y_col="level",
		target_size=(512, 512),
		batch_size=batch_size,
		class_mode='categorical'
		)


	val_generator = train_gen.flow_from_dataframe(
			dataframe=test,
		directory='/media/bmi/poseidon/DiabeticR/train_cropped',
		x_col="image",
		y_col="level",
		target_size=(512, 512),
		batch_size=batch_size,
		class_mode='categorical'
		)

	def plotImages(images_arr):
	    fig, axes = plt.subplots(1, 5, figsize=(20,20))
	    axes = axes.flatten()
	    for img, ax in zip( images_arr, axes):
	        ax.imshow(img/np.max(img))
	    plt.tight_layout()
	    plt.show()
	    
	    
	augmented_images = [train_generator[0][0][0] for i in range(5)]
	print(augmented_images[0], np.ptp(augmented_images[0]//255.))
	#plotImages(augmented_images)

	# # this is the model we will train
	model = create_model(np.ones(5))

	T = Training(model, nb_epoch = 1, batch_size=batch_size, savepath='/media/bmi/poseidon/DiabeticR/Resnet101.hdf5')

	T.fit(train_generator, val_generator)

	class_weights = class_weight.compute_class_weight('balanced',
	                                         np.unique(y_train),
	                                         y_train)

	class_weights_clipped_1 = np.clip(class_weights, 0, 0.8)

	class_weights_clipped_2 = np.clip(class_weights, 0, 2)

	# this is the model we will train
	model = create_model(class_weights_clipped_1)

	T = Training(model, nb_epoch = 10, batch_size=batch_size, load_model_resume_training=True,
	 weight_path='/media/bmi/poseidon/DiabeticR/Resnet101.hdf5', savepath = '/media/bmi/poseidon/DiabeticR/Resnet101_P2.hdf5')
	T.fit(train_generator, val_generator)

	model = create_model(class_weights_clipped_2)

	T = Training(model, nb_epoch = 5, batch_size=batch_size, load_model_resume_training=True,
	 weight_path='/media/bmi/poseidon/DiabeticR/Resnet101_P2.hdf5', savepath = '/media/bmi/poseidon/DiabeticR/Resnet101_P3.hdf5')
	T.fit(train_generator, val_generator)
