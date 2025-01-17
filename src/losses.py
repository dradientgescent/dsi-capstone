from keras import backend as K
import tensorflow as tf



def categorical_focal_loss(gamma=2., alpha = 0.25):
		"""
		Softmax version of focal loss.
					 m
			FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
					c=1
			where m = number of classes, c = class and o = observation
		Parameters:
			alpha -- the same as weighing factor in balanced cross entropy
			gamma -- focusing parameter for modulating factor (1-p)
		Default value:
			gamma -- 2.0 as mentioned in the paper
			alpha -- 0.25 as mentioned in the paper
		References:
				Official paper: https://arxiv.org/pdf/1708.02002.pdf
				https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
		Usage:
		 model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
		"""
		def categorical_focal_loss_fixed(y_true, y_pred):
				"""
				:param y_true: A tensor of the same shape as `y_pred`
				:param y_pred: A tensor resulting from a softmax
				:return: Output tensor.
				"""

				# Scale predictions so that the class probas of each sample sum to 1
				y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

				# Clip the prediction value to prevent NaN's and Inf's
				epsilon = K.epsilon()
				y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

				# Calculate Cross Entropy
				cross_entropy = -y_true * K.log(y_pred)

				# Calculate Focal Loss
				loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

				# Sum the losses in mini_batch
				return K.sum(loss, axis=1)

		return categorical_focal_loss_fixed

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
