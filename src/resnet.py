from keras_applications.inception_v3 import InceptionV3
from keras_applications.resnet import ResNet101
from keras.callbacks import  ModelCheckpoint
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.models import model_from_json,load_model
from dataloader import DataGenerator
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from keras import backend as K
import pandas as pd
import numpy as np

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
	base_model = ResNet101(weights='imagenet', include_top=False, backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)

	# add a global spatial average pooling layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	# let's add a fully-connected layer
	x = Dense(1024, activation='relu')(x)
	x = Dense(256, activation='relu')(x)
	# and a logistic layer -- let's say we have 200 classes
	predictions = Dense(5, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=predictions)

	sgd = SGD(lr=0.01, momentum=0.9, decay=5e-6, nesterov=False)
	model.compile(optimizer=sgd, loss=weighted_categorical_crossentropy(class_weights))

	return model


class Training(object):
    

    def __init__(self, model, nb_epoch, load_model_resume_training=None):

        
        self.nb_epoch = nb_epoch


        #loading model from path to resume previous training without recompiling the whole model
        if load_model_resume_training is not None:
            self.model =load_model(load_model_resume_training)
            print("pre-trained model loaded!")
        else:
            self.model = model
            #self.model.load_weights('/home/parth/Interpretable_ML/Brain-tumor-segmentation/checkpoints/Unet_cc/SimUnet.01_0.095.hdf5')
            print("Model compiled!")

    def fit(self, train_gen, val_gen):
		
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

        train_generator = train_gen
        val_generator = val_gen
        checkpointer = ModelCheckpoint(filepath='/media/bmi/poseidon/DiabeticR/models.{epoch:02d}_{val_loss:.3f}.hdf5', verbose=1, period = 5)
        self.model.fit_generator(train_generator,
                                 epochs=self.nb_epoch, validation_data=val_generator,  verbose=1,
                                 callbacks=[checkpointer, reduce_lr], class_weight = class_weights)



if __name__ == '__main__':

	label_df = pd.read_csv('/media/bmi/poseidon/DiabeticR/trainLabels.csv')

	y_train = np.array(label_df['level'])

	class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)

	class_weights = np.clip(class_weights, 0, 3)
	
	# this is the model we will train
	model = create_model(class_weights)

	train_generator = DataGenerator('/media/bmi/poseidon/DiabeticR/train_resized/', batch_size = 16)
	val_generator = DataGenerator('/media/bmi/poseidon/DiabeticR/val_resized/', batch_size = 16)

	# train_gen = ImageDataGenerator(
	#     rotation_range=10,
	#     width_shift_range=10,
	#     height_shift_range=10,
	#     horizontal_flip=True)

	# val_gen = ImageDataGenerator()

	# train_generator = train_gen.flow_from_directory(
 #        '/media/parth/DATA/DiabeticR/train_resized',
 #        target_size=(256, 256),
 #        batch_size=16,
 #        class_mode='categorical')

	# val_generator = val_gen.flow_from_directory(
 #        '/media/parth/DATA/DiabeticR/train_resized',
 #        target_size=(256, 256),
 #        batch_size=16,
 #        class_mode='categorical')

	T = Training(model, nb_epoch = 100)

	T.fit(train_generator, val_generator)
