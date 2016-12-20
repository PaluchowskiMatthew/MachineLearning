from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import SGD
from keras import backend as K

from image_handling import *

NUMBER_IMGS = 50
TRAIN_RATIO = 0.8
NUM_CHANNELS = 3

batch_size = 128
nb_classes = 2


# ********** Tuning parameters: (See Network architecture as well)

# size of patch of an image to be used as input and output of the neural net
IMG_PATCH_SIZE = 8
# Epochs to be trained
nb_epoch = 12
# number of convolutional filters to use
nb_filters_layer1 = 64
nb_filters_layer2 = 128
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size_layer1 = (5, 5)
kernel_size_layer2 = (3, 3)

input_shape = (IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS)

def train_cnn(model_name='dummy.h5'):

	# ***************** HANDLE THE DATA **********************************
	data_dir = 'training/'
	train_data_filename = data_dir + 'images/'
	train_labels_filename = data_dir + 'groundtruth/' 

	# Extract data into numpy arrays.
	data = extract_data(train_data_filename, NUMBER_IMGS, IMG_PATCH_SIZE)
	labels = extract_labels(train_labels_filename, NUMBER_IMGS, IMG_PATCH_SIZE)

	# Create train and test sets
	idx = np.random.permutation(np.arange(data.shape[0]))
	train_size = int(TRAIN_RATIO*data.shape[0])
	X_train = data[idx[:train_size]]
	Y_train = labels[idx[:train_size]]
	X_test = data[idx[train_size:]]
	Y_test = labels[idx[train_size:]]
	
	# **************** DEFINE THE MODEL ARCHITECTURE *******************

	model = Sequential()

	# Convolution layer with rectified linear activation
	model.add(Convolution2D(nb_filters_layer1, kernel_size_layer2[0], 
							kernel_size_layer2[1], border_mode='same',
							input_shape=input_shape))
	model.add(Activation('relu'))

	# Second convolution
	model.add(Convolution2D(nb_filters_layer2, kernel_size_layer2[0], 
							kernel_size_layer2[1]))
	model.add(Activation('relu'))

	# Third convolution
	model.add(Convolution2D(nb_filters_layer1, kernel_size_layer1[0], 
							kernel_size_layer2[1]))
	model.add(Activation('relu'))

	# Pooling and dropout
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(SpatialDropout2D(0.25))

	# Full-connected layer
	model.add(Flatten())

	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(Dense(512))
	model.add(Activation('relu'))

	# Dropout to avoid overfitting
	#model.add(Dropout(0.25))

	#model.add(Dense(4096))
	#model.add(Activation('relu'))

	#model.add(Dense(512))
	#model.add(Activation('relu'))

	# Dropout to avoid overfitting
	model.add(Dropout(0.5))

	#Fully-connected layer to ouptut the resulting class
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	model.compile(loss='binary_crossentropy',
			  optimizer='adadelta',
			  metrics=['fmeasure'])

	#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	#model.compile(loss='binary_crossentropy',
	#			  optimizer=sgd,
	#			  metrics=['fmeasure'])


	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
			  class_weight='auto', verbose=1, validation_data=(X_test, Y_test))


	score = model.evaluate(X_test, Y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	model.save('models/' + model_name)