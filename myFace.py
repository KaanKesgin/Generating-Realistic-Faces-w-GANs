#GANS for face generation
import os
import helper
from glob import glob
from matplotlib import pyplot

import warnings
import tensorflow as tf

import numpy as np


#check for GPU because why not
#if not tf.test.gpu_device_name():
#	warnings.warn('No GPU found, You are going to want to use a GPU for this')
#else:
#	print('Default GPU device: {}'.format(tf.test.gpu_device_name))

data_dir = 'data'
show_n_images = 25

helper.download_extract('mnist', data_dir)
#helper.download_extract('celeba', data_dir)

mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'mnist/*.jpg'))[:show_n_images], 28, 28, 'L')
pyplot.imshow(helper.images_square_grid(mnist_images, 'L'), cmap='gray')	

def model_inputs(image_width, image_height, image_channels, z_dim):
	""""create model_inputs
	:param image_width: 	the input image width
	:param image _height: 	input image height
	:param image_channels:	the number of image channels
	:param z_dim: 			the dimension of Z 

	:return: 				Tuple of (tensor of real input images, tensor of z data, learning rate)
	"""
	realInputImages = tf.placeholder(tf.float32, (None,image_width,image_height,image_channels), name='realInputImages')
	inputs_Z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
	learning_rate = tf.placeholder(tf.float32)

	return realInputImages, inputs_Z, learning_rate


#Discriminator network that discriminates on images
#This function should be able to reuse variables in the neural network, use a scope name of discriminator
#The function should return a tuple of (tensor output of the discriminator, tensor logits of the discriminator)

def discriminator(images, reuse=False):
	"""Create the discriminator network
	:param images: 	tensor of input image(s)
	:param reuse: 	boolean if the weights should be reused
	:return: 		tuple of (tensor output of the discriminator, tensor logits of the discriminator)"""

	alpha = 0.2

	with tf.variable_scope('discriminator', reuse=reuse):
		#input layer is 28*28*3(?!_1)
		x1 = tf.layers.conv2d(images, 64, 5,strides=2, padding='same')
		# 64 is filter: dimensionality of the output space (number of filters in the convolution)
		# 5 is the stride of convolution along the height and width (can be a tuple with 2 dimensions)
		relu1 = tf.maximum(alpha*x1, x1) 
		# dimensions are now 14*14*64
		x2 = tf.layers.conv2d(relu1, 128, 5, strides=2, padding='same')
		bn2 = tf.layers.batch_normalization(x2, training=True)
		relu2 = tf.maximum(alpha*bn2, bn2)
		# dimensions are now 7*7*128
		# time to flatten
		flat = tf.reshape(relu2, (-1, 7*7*128))
		logits = tf.layers.dense(flat,1)
		out = tf.sigmoid(logits)

		return out, logits

#implement a generator to generate an image using 'z'. 
#similar to discriminator this function should be able to reuse the variables in the neural network
#in a similar manner with the scope defined in the discriminator network
#this function should return the generated 28*28*out_channel_dim images
def generator(z, out_channel_dim, is_train=True):
	"""Create the generator network
	:param z: 					Input z
	:param out_channel_dim: 	the number of channels in the output image
	:param: is_train: 			boolean if generator is being used for training

	:return: 					the tensor output of the generator"""
	alpha = 0.2
	with tf.variable_scope('generator', reuse=not is_train):
		#first fully connected layer
		x1 = tf.layers.dense(z, 7*7*128)
		# reshape it to start the convolutional stack
		x1 = tf.reshape(x1, (-1, 7, 7, 128))
		x1 = tf.layers.batch_normalization(x1, training=is_train)
		x1 = tf.maximum(alpha*x1, x1)
		#current size 7*7*128
		x2 = tf.layers.conv2d_transpose(x1, 64, 5, strides=2, padding='same')
		x2 = tf.layers.batch_normalization(x2, training=is_train)
		x2 = tf.maximum(alpha * x2, x2)
		#current size is 14*14*64
		logits = tf.layers.conv2d_transpose(x2, out_channel_dim, 5, strides=2, padding='same')
		#current size = 28*28*3
		out = tf.tanh(logits)
		return out


#implement model_loss to build the GANs for training and calculate the loss. 
#The functions should return a tuple of (discriminator loss, generator loss)
#use discriminator and generator functions

def model_loss(input_real, input_z, out_channel_dim):
	"""Get the loss for the discriminator and generator
	:param input_real: Images from the real dataset
	:param input_z: Z inpput
	:param out_channel_dim: the number of channels in the output image

	:return: a tuple of (discriminator loss, generator loss)"""

	g_model = generator(input_z, out_channel_dim)
	d_model_real, d_logits_real = discriminator(input_real)
	d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)

	d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
																		labels=tf.ones_like(d_model_real)))
	d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
																		labels=tf.zeros_like(d_model_fake)))
	g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
																		labels=tf.ones_like(d_model_fake)))
	d_loss = d_loss_real + d_loss_fake

	return d_loss, g_loss


#implement model_opt to create the optimization operations for the GAN
#the function should return a tuple of (discriminator training operatrion, generator treining operation)
def model_opt(d_loss, g_loss, learning_rate, beta1):
	"""Get optimization operations 
	:param d_loss: 			Discriminator loss tensor
	:param g_loss: 			Generator loss tensor
	:param learning_rate: 	learning rate placeholder
	:param beta1: 			the exponential decay rate for the 1st moment in the optimizer

	:return: A tuple of(discriminator training operation, generator training operation)
	"""
	
	t_vars = tf.trainable_variables()
	d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
	g_vars = [var for var in t_vars if var.name.startswith('generator')]

	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
		g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

		return d_train_opt, g_train_opt



#following function is to show the output of the generator during training to help determine how well GAN is training
def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
	"""show example output for the generator
	:param sess: 				tensorflow session
	:param n_images: 			number of images to display
	:param input_z: 			input Z tensor
	:param out_channel_dim: 	the number of channels in the output image
	:param image_mode: 			the mode to use for images ("RGB" or "L)"""

	cmap = None if image_mode == 'RGB' else 'gray'
	z_dim = input_z.get_shape().as_list()[-1]
	example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

	samples = sess.run(generator(input_z, out_channel_dim, False), feed_dict={input_z: example_z})

	images_grid = helper.images_square_grid(samples, image_mode)
	pyplot.imshow(images_grid, cmap=cmap)

#implementation of train to build and train GANs
#its recommended to show generator output once every 100 batches as it is a computational expense to show the output


def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
	"""Train the GAN
	:param epoch_count: number of epochs
	:param batch_size: batch size
	:param z_dim: Z dimension
	:param learning _rate: learning rate
	:param beta1: exponential decay rate for the 1st moment in the optimizer
	:param get_batches: function to get batches
	:param data_shape: shape of the data
	:param data_image_mode the image mode ("RGB" or "L")"""

	inputs_real, inputs_z, learning_rate = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
	learnRate = 0.0002

	d_loss, g_loss = model_loss(inputs_real, inputs_z, data_shape[3])
	d_opt, g_opt = model_opt(d_loss, g_loss, learnRate, beta1)

	step = 0
	losses = []

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch_i in range(epoch_count):
			for batch_images in get_batches(batch_size):
				step +=1
				batch_images *=2
				batch_z = np.random.uniform(-1,1,size=(batch_size, z_dim))
				sess.run(d_opt, feed_dict={inputs_real: batch_images, inputs_z: batch_z, learning_rate: learnRate})
				sess.run(g_opt, feed_dict={inputs_z: batch_z, inputs_real: batch_images, learning_rate: learnRate})

				if step%128 == 0:
					train_loss_d = d_loss.eval({inputs_z:batch_z, inputs_real: batch_images})
					train_loss_g = g_loss.eval({inputs_z: batch_z})
					print("Epoch {}/{}...".format(epoch_i+1, epochs), 
							"Discriminator Loss: {:.4f} ...".format(train_loss_d),
							"Generator Loss: {:.4f} ...".format(train_loss_g))

					losses.append((train_loss_d, train_loss_g))

					show_generator_output(sess, 16, inputs_z, data_shape[3], data_image_mode)




batch_size = 32
z_dim = 100
learning_rate = 0.0002
beta1 = 0.1

epochs = 1
celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
with tf.Graph().as_default():
	train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
		  celeba_dataset.shape, celeba_dataset.image_mode)




































