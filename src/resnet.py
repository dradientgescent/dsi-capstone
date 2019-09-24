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

def create_model():

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
	model.compile(optimizer=sgd, loss='categorical_crossentropy')

	return model


class Training(object):
    

    def __init__(self, model, nb_epoch,load_model_resume_training=None):

        
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

    	reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

        train_generator = train_gen
        val_generator = val_gen
        checkpointer = ModelCheckpoint(filepath='/media/bmi/poseidon/DiabeticR/models.{epoch:02d}_{val_loss:.3f}.hdf5', verbose=1, period = 5)
        self.model.fit_generator(train_generator,
                                 epochs=self.nb_epoch, validation_data=val_generator,  verbose=1,
                                 callbacks=[checkpointer, reduce_lr])



if __name__ == '__main__':

	# this is the model we will train
	model = create_model()

	train_generator = DataGenerator('/media/bmi/poseidon/DiabeticR/train_resized/', batch_size = 16)
	val_generator = DataGenerator('/media/bmi/poseidon/DiabeticR/val_resized/', batch_size = 16)

	T = Training(model, nb_epoch = 100)

	T.fit(train_generator, val_generator)
