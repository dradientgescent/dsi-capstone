from __future__ import absolute_import, division, print_function, unicode_literals


try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

import numpy as np
import time
import functools
import time
import cv2


def load_img(path_to_img):
	max_dim = 512
	img = tf.io.read_file(path_to_img)
	img = tf.image.decode_image(img, channels=3)
	img = tf.image.convert_image_dtype(img, tf.float32)

	shape = tf.cast(tf.shape(img)[:-1], tf.float32)
	long_dim = max(shape)
	scale = max_dim / long_dim

	new_shape = tf.cast(shape * scale, tf.int32)

	img = tf.image.resize(img, new_shape)
	img = img[tf.newaxis, :]
	return img

class Styler():

	def __init__(self, model, content_image, style_image, content_layers, style_layers):

		self.model = model
		self.content_image = content_image
		self.style_image = style_image
		self.content_layers = content_layers
		self.style_layers = style_layers
		self.num_style_layers = len(style_layers)
		self.num_content_layers = len(content_layers)
		self.model.trainable = False

	def imshow(self, image, title=None):
		if len(image.shape) > 3:
			image = tf.squeeze(image, axis=0)

		plt.imshow(image)
		if title:
			plt.title(title)


	def model_layers(self, model, layer_names):
		""" Creates a model that returns a list of intermediate output values."""
		# Load our model. Load pretrained VGG, trained on imagenet data
		style_model = self.model
		style_model.trainable = False

		outputs = [style_model.get_layer(name).output for name in layer_names]

		model = tf.keras.Model([style_model.input], outputs)
		return model

	@staticmethod
	def gram_matrix(input_tensor):
	  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
	  input_shape = tf.shape(input_tensor)
	  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
	  return result/(num_locations)

	def StyleContentModel(self, inputs):

		style_model = self.model_layers(self.model, self.style_layers+self.content_layers)

	    inputs = inputs*255.0
	    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
	    outputs = style_model(preprocessed_input)
	    style_outputs, content_outputs = (outputs[:self.num_style_layers], 
	                                      outputs[self.num_style_layers:])

	    style_outputs = [gram_matrix(style_output)
	                     for style_output in style_outputs]

	    content_dict = {content_name:value 
	                    for content_name, value 
	                    in zip(self.content_layers, content_outputs)}

	    style_dict = {style_name:value
	                  for style_name, value
	                  in zip(self.style_layers, style_outputs)}
	    
	    return {'content':content_dict, 'style':style_dict}


	def style_content_loss(self, outputs, style_targets, content_targets):

		style_outputs = outputs['style']
		content_outputs = outputs['content']

		style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
		                       for name in style_outputs.keys()])
		style_loss *= style_weight / self.num_style_layers

		content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
		                         for name in content_outputs.keys()])
		content_loss *= content_weight / self.num_content_layers
		loss = style_loss + content_loss
		return loss



if __name__ == '__main__':

	# Load images
	content_path = tf.keras.utils.get_file('turtle.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg')
	content_image = load_img(self.content_path)
	style_image = cv2.imread('/content/drive/My Drive/Colab Notebooks/pebbles-29.jpg')
	style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)

	plt.subplot(1, 2, 1)
	imshow(content_image, 'Content Image')

	plt.subplot(1, 2, 2)
	imshow(style_image, 'Style Image')

	vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

	# Content layer where will pull our feature maps
	content_layers = ['block5_conv2'] 

	# Style layer of interest
	style_layers = ['block1_conv1',
	                'block2_conv1',
	                'block3_conv1', 
	                'block4_conv1', 
	                'block5_conv1']

	S = Styler(vgg, content_image, style_image, content_layers, style_layers)

	# results = StyleContentModel(tf.constant(content_image))

	# style_results = results['style']

	
	style_image = cv2.resize(style_image, (512, 512))
	style_targets = StyleContentModel(style_image[None, ...]/255.0)['style']
	content_targets = StyleContentModel(content_image)['content']

	image = tf.Variable(content_image)

	def clip_0_1(image):
	  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

	opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

	style_weight=1e-2
	content_weight=1e4

	@tf.function()
	def train_step(image):
		with tf.GradientTape() as tape:
			outputs = StyleContentModel(image)
			loss = style_content_loss(outputs, style_targets, content_targets)

		grad = tape.gradient(loss, image)
		opt.apply_gradients([(grad, image)])
		image.assign(clip_0_1(image))

	start = time.time()

	epochs = 10
	steps_per_epoch = 100

	step = 0
	for n in range(epochs):
	  for m in range(steps_per_epoch):
	    step += 1
	    train_step(image)
	    print(".", end='')
	  display.clear_output(wait=True)
	  imshow(image.read_value())
	  plt.title("Train step: {}".format(step))
	  plt.show()

	end = time.time()
	print("Total time: {:.1f}".format(end-start))
