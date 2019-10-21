from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
from keras import layers
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
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
from tqdm import tqdm
from enet_preprocess import *
from keras_radam import RAdam
from group_norm import GroupNormalization
sys.path.append(os.path.abspath('../../efficientnet/'))
import efficientnet.keras as efn 
from efficientnet.model import EfficientNetB5
from dataloader import DataGenerator



def create_model():

	IMG_WIDTH, IMG_HEIGHT, CHANNELS = 460, 460, 3
	elu = keras.layers.ELU(alpha=1.0)

	# create the base pre-trained model
	# Load in EfficientNetB5
	effnet = efn.EfficientNetB5(weights=None,
	                        include_top=False,
	                        input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS))
	effnet.load_weights('/media/brats/mirlproject2/aptos_2019/efficientnet-b5_imagenet_1000_notop.h5')

	# Replace all Batch Normalization layers by Group Normalization layers
	for i, layer in enumerate(effnet.layers):
	    if "batch_normalization" in layer.name:
	        effnet.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)

	model = Sequential()
	model.add(effnet)
	model.add(GlobalAveragePooling2D())
	model.add(Dropout(0.5))
	model.add(Dense(5, activation=elu))
	model.add(Dense(1, activation="linear"))
	model.compile(loss='mse',
	              optimizer=RAdam(lr=0.00005), 
	              metrics=['mse', 'acc'])
	print(model.summary())

	return model

def get_preds_and_labels(model, generator):
    """
    Get predictions and labels from the generator
    """
    preds = []
    labels = []
    for _ in range(int(np.ceil(generator.samples / BATCH_SIZE))):
        x, y = next(generator)
        preds.append(model.predict(x))
        labels.append(y)
    # Flatten list of numpy arrays
    return np.concatenate(preds).ravel(), np.concatenate(labels).ravel()

class Metrics(Callback):
	"""
	A custom Keras callback for saving the best model
	according to the Quadratic Weighted Kappa (QWK) metric
	"""
	SAVED_MODEL_NAME = 'effnet.h5'
	def on_train_begin(self, logs={}):
	    """
	    Initialize list of QWK scores on validation data
	    """
	    self.val_kappas = []

	def on_epoch_end(self, epoch, logs={}):
	    """
	    Gets QWK score on the validation data
	    """
	    # Get predictions and convert to integers
	    y_pred, labels = get_preds_and_labels(model, val_generator)
	    y_pred = np.rint(y_pred).astype(np.uint8).clip(0, 4)
	    # We can use sklearns implementation of QWK straight out of the box
	    # as long as we specify weights as 'quadratic'
	    _val_kappa = cohen_kappa_score(labels, y_pred, weights='quadratic')
	    self.val_kappas.append(_val_kappa)
	    print("val_kappa: {}".format(round(_val_kappa, 4)))
	    if _val_kappa == max(self.val_kappas):
	        print("Validation Kappa has improved. Saving model.")
	        self.model.save(SAVED_MODEL_NAME)
	    return



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

		# For tracking Quadratic Weighted Kappa score
		kappa_metrics = Metrics()
		# Monitor MSE to avoid overfitting and save best model
		es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=12)
		rlr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
								factor=0.5, 
								patience=4, 
								verbose=1, 
								mode='auto', 
								epsilon=0.0001)

		train_generator = train_gen
		val_generator = val_gen
		checkpointer = ModelCheckpoint(filepath=self.savepath, verbose=1, save_best_only = True)
		self.model.fit_generator(train_generator,
										 epochs=self.nb_epoch, validation_data=val_generator, 
										 steps_per_epoch = len(train_generator), validation_steps = len(val_generator),  
										 verbose=1, callbacks=[kappa_metrics, rlr, es])



if __name__ == '__main__':

	label_df = pd.read_csv('/media/brats/mirlproject2/aptos_2019/train.csv')

	#label_df["id_code"] = label_df["id_code"].apply(lambda name : name + '.png')

	# label_df["level"] = label_df["level"].apply(lambda label : str(label))

	train, test = train_test_split(label_df, test_size=0.25, random_state=42)

	print(train.shape, test.shape)

	y_train = np.array(label_df['diagnosis'])

	sample_array = []	

	BATCH_SIZE = 4
	IMG_WIDTH, IMG_HEIGHT = 460, 460
	TRAIN_IMG_PATH = '/media/brats/mirlproject2/aptos_2019/train_cropped'

	# Add Image augmentation to our generator
	train_datagen = DataGenerator(TRAIN_IMG_PATH, batch_size = 8, dataframe = train)

	val_datagen = DataGenerator(TRAIN_IMG_PATH, batch_size = 8, dataframe = test)
	# print('\n\n')
	# print(len(train_datagen))
	# model = create_model()

	#T = Training(model, nb_epoch = 10, batch_size = 8, savepath = 'effnet.h5', load_model_resume_training=None, weight_path=None)

	#T.fit(train_datagen, val_datagen)

	# val_gen = ImageDataGenerator(
	# 	samplewise_center=True,
	# 	samplewise_std_normalization=True
	# 	)

	# # Use the dataframe to define train and validation generators
	# train_generator = train_datagen.flow_from_dataframe(label_df, 
	#                                                     x_col='id_code', 
	#                                                     y_col='diagnosis',
	#                                                     directory = TRAIN_IMG_PATH,
	#                                                     target_size=(IMG_WIDTH, IMG_HEIGHT),
	#                                                     batch_size=BATCH_SIZE,
	#                                                     class_mode='other', 
	#                                                     color_mode='rgb',
	#                                                     subset='training')

	# val_generator = train_datagen.flow_from_dataframe(label_df, 
	#                                                   x_col='id_code', 
	#                                                   y_col='diagnosis',
	#                                                   directory = TRAIN_IMG_PATH,
	#                                                   target_size=(IMG_WIDTH, IMG_HEIGHT),
	#                                                   batch_size=BATCH_SIZE,
	#                                                   class_mode='other',
	#                                                   subset='validation')


	# def plotImages(images_arr):
	#     fig, axes = plt.subplots(1, 5, figsize=(20,20))
	#     axes = axes.flatten()
	#     for img, ax in zip( images_arr, axes):
	#         ax.imshow(img)
	#     plt.tight_layout()
	#     plt.show()
	print(1)
	plt.imshow(train_datagen.__getitem__(0)[0][0])
	plt.show()
	# print(img)
	# plt.imshow(img)
	# plt.show()
	#plotImages(augmented_images)
