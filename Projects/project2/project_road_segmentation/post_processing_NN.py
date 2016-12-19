from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import SGD
from keras import backend as K

from image_handling import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

IMG_SIZE = 400
PATCH_INPUT = 10
PATCH_UNIT = 8
nb_class = 2

PATCH_WINDOW = 5  # MUST BE ODD

batch_size = 50
nb_epoch = 12

def post_process(model_name, train_range):

	#******* EXCTRACT DATA ********************
	data_dir = 'training/'
	train_data_filename = data_dir + 'images/'
	train_labels_filename = data_dir + 'groundtruth/'

	pred = extract_data_post(model_name, train_range, train_data_filename, PATCH_UNIT, PATCH_WINDOW)
	Y = extract_labels_post(train_range, train_labels_filename, PATCH_UNIT, PATCH_WINDOW)

	#******** CHECK ACCURACY OF PREDICTIONS ****
	new_size = int(IMG_SIZE/PATCH_UNIT)
	w = int((PATCH_WINDOW-1)/2)
	size_tr = int(new_size - 2*w)
	num_images = train_range[1]-train_range[0]+1
	TP=0.
	FN=0.
	for ii in np.arange((num_images*size_tr**2)):
		if pred[ii, 2, 2] == Y[ii, 0]:
			TP = TP+1.
		else:
			FN = FN +1.
	print("Acc: ")
	print(TP/(TP+FN))

	return
	# ************* TRAIN AND TEST SETS *******
	TRAIN_RATIO = 0.7
	idx = np.random.permutation(np.arange(pred.shape[0]))
	train_size = int(TRAIN_RATIO*pred.shape[0])
	X_train = pred[idx[:train_size]]
	Y_train = Y[idx[:train_size]]
	X_test = pred[idx[train_size:]]
	Y_test = Y[idx[train_size:]]

	# ************* NEURAL NET *****************
	input_shape = (PATCH_WINDOW, PATCH_WINDOW, 1)
	model2 = Sequential()
	# Convolution layer with rectified linear activation
	model2.add(Convolution2D(64, 3,3, border_mode='same',
							input_shape=input_shape))
	model2.add(Activation('relu'))

	model2.add(Convolution2D(64, 3,3))
	model2.add(Activation('relu'))

	#model2.add(MaxPooling2D(pool_size=(2, 2)))
	model2.add(Flatten())
	model2.add(Dense(512))
	model2.add(Dropout(0.25))
	model2.add(Activation('relu'))
	#model2.add(Dense(output_dim=(PATCH_INPUT*PATCH_INPUT, nb_class) , activation='softmax'))
	model2.add(Dense(nb_class))
	model2.add(Activation('softmax'))

	model2.compile(loss='categorical_crossentropy',
			  optimizer='adadelta',
			  metrics=['fmeasure'])


	#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	#model.compile(loss='binary_crossentropy',
	#			  optimizer=sgd,
	#			  metrics=['fmeasure'])


	model2.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
			  class_weight='auto', verbose=1, validation_data=(X_test, Y_test))


	score = model2.evaluate(X_test, Y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
