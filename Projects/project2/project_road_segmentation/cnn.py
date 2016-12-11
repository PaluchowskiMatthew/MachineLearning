from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import SGD
from keras import backend as K

from image_handling import *

NUMBER_IMGS = 100
TRAIN_RATIO = 0.8
NUM_CHANNELS = 3

batch_size = 128
nb_classes = 2


# ********** Tuning parameters: (See Network architecture as well)

# size of patch of an image to be used as input and output of the neural net
IMG_PATCH_SIZE = 8
# Epochs to be trained
nb_epoch = 20
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (4, 3)

input_shape = (IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS)

def train_cnn(model_name='dummy.h5'):

	# ***************** HANDLE THE DATA **********************************
	data_dir = 'training/'
	train_data_filename = data_dir + 'images/'
	train_labels_filename = data_dir + 'groundtruth/' 

	# Extract data into numpy arrays.
	data = extract_data(train_data_filename, NUMBER_IMGS, IMG_PATCH_SIZE)
	labels = extract_labels(train_labels_filename, NUMBER_IMGS, IMG_PATCH_SIZE)
	#print(train_data.shape)
	#print(train_labels.shape)

	# Create train and test sets
	idx = np.random.permutation(np.arange(data.shape[0]))
	train_size = int(TRAIN_RATIO*data.shape[0])
	X_train = data[idx[:train_size]]
	Y_train = labels[idx[:train_size]]
	X_test = data[idx[train_size:]]
	Y_test = labels[idx[train_size:]]

	"""
	# Balancing the class VS. class_weight during traing?
	c0 = 0
	c1 = 0
	for i in range(len(Y_train)):
		if Y_train[i][0] == 1:
			c0 = c0 + 1
		else:
			c1 = c1 + 1
	print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

	print ('Balancing training data...')
	min_c = min(c0, c1)
	idx0 = [i for i, j in enumerate(Y_train) if j[0] == 1]
	idx1 = [i for i, j in enumerate(Y_train) if j[1] == 1]
	new_indices = idx0[0:min_c] + idx1[0:min_c]
	print (len(new_indices))
	print (Y_train.shape)
	X_train = X_train[new_indices,:,:,:]
	Y_train = Y_train[new_indices]

	train_size = Y_train.shape[0]

	c0 = 0
	c1 = 0
	for i in range(len(Y_train)):
		if Y_train[i][0] == 1:
			c0 = c0 + 1
		else:
			c1 = c1 + 1
	print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

	"""
	# **************** DEFINE THE MODEL ARCHITECTURE *******************

	model = Sequential()

	# Convolution layer with rectified linear activation
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
							border_mode='same',
							input_shape=input_shape))
	model.add(Activation('relu'))

	# Second convolution
	model.add(Convolution2D(nb_filters, kernel_size[1], kernel_size[0]))
	model.add(Activation('relu'))

	model.add(Dropout(0.25))

	# Third convolution
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[0]))
	model.add(Activation('relu'))

	# Pooling and dropout
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(0.25))

	# Full-connected layer
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('relu'))

	model.add(Dense(1024))
	model.add(Activation('relu'))

	# Dropout to avoid overfitting
	model.add(Dropout(0.25))

	model.add(Dense(1024))
	model.add(Activation('relu'))

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

	#class_weight = auto??
	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
			  class_weight='auto', verbose=1, validation_data=(X_test, Y_test))


	score = model.evaluate(X_test, Y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	model.save('models/' + model_name)

	"""
	data_dir = 'test_set_images/'
	pred_dir = 'predictions/'
	for i in range(1, 51):
		imageid = "test_%.1d" % i
		image_filename = data_dir + imageid + ".png"
		if os.path.isfile(image_filename):
			print ('Predicting' + image_filename)
			img = mpimg.imread(image_filename)

			data = np.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))

			predictions_patch = model.predict_classes(data, verbose=1)

			img_prediction = label_to_img(img.shape[0], img.shape[1], 
										  IMG_PATCH_SIZE, IMG_PATCH_SIZE, 
										  predictions_patch)

			pimg = Image.fromarray((img_prediction*255.0).astype(np.uint8))
			pimg.save(pred_dir + "prediction_" + str(i) + ".png")

		else:
			print ('File ' + image_filename + ' does not exist')
	"""