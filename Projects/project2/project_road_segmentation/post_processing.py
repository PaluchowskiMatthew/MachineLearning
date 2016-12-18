from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import SGD
from keras import backend as K

from image_handling import *

IMG_SIZE = 400
PATCH_INPUT = 10
PATCH_UNIT = 8
nb_class = 2

batch_size = 50
nb_epoch = 12

def post_process(model, train_range):

	# Load the trained model
	#model_path = 'models/' + model_name
	#model = load_model(model_path)
	#model.compile(loss='categorical_crossentropy',
				   #optimizer='adadelta',
				   #metrics=['fmeasure'])

	data_dir = 'training/'
	train_data_filename = data_dir + 'images/'
	train_labels_filename = data_dir + 'groundtruth/'

	#EXTRACT LABELS
	gt_imgs = []
	for i in range(train_range[0], train_range[1]+1):
		imageid = "satImage_%.3d" % i
		image_filename = train_labels_filename + imageid + ".png"
		if os.path.isfile(image_filename):
			print ('Loading ' + image_filename)
			img = mpimg.imread(image_filename)
			gt_imgs.append(img)
		else:
			print ('File ' + image_filename + ' does not exist')
	num_images = len(gt_imgs)
	gt_patches = [img_crop(gt_imgs[i], PATCH_UNIT, PATCH_UNIT) for i in range(num_images)]
	#data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
	
	new_size = int(IMG_SIZE/PATCH_UNIT)
	imgs = np.zeros((num_images, new_size, new_size, nb_class))
	for i in range(num_images):
		patch_img = np.asarray([value_to_class(np.mean(np.asarray(gt_patches[i][j]))) 
			                    for j in range(len(gt_patches[i]))])
		imgs[i, :, :, :] = np.reshape(patch_img, (new_size, new_size, nb_class))
	img_patches = [img_crop(imgs[i], PATCH_INPUT, PATCH_INPUT) for i in range(num_images)]
	labels = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
	#labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])
	labels = np.reshape(labels, (labels.shape[0], labels.shape[1]*labels.shape[2], labels.shape[3]))
	Y = labels.astype(np.float32)

	imgs = np.zeros((num_images, new_size, new_size, 1))
	j=0

	#Extract images
	for i in range(train_range[0], train_range[1]+1):
		imageid = "satImage_%.3d" % i
		image_filename = train_data_filename + imageid + ".png"
		if os.path.isfile(image_filename):
			print ('Predicting' + image_filename)
			img = mpimg.imread(image_filename)
			data = np.asarray(img_crop(img, PATCH_UNIT, PATCH_UNIT))
			predictions_patch = model.predict_classes(data, verbose=1)
			imgs[j, :, :, :] = np.reshape(predictions_patch, (new_size, new_size, 1))
			j+1
		else:
			print ('File ' + image_filename + ' does not exist')

	img_patches = np.asarray([img_crop(imgs[i], PATCH_INPUT, PATCH_INPUT) for i in range(num_images)])
	data = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
	#datashape: 275*10*10
	print(data.shape)
	print(Y.shape)

	TRAIN_RATIO = 0.8
	idx = np.random.permutation(np.arange(data.shape[0]))
	train_size = int(TRAIN_RATIO*data.shape[0])
	X_train = data[idx[:train_size]]
	Y_train = Y[idx[:train_size]]
	X_test = data[idx[train_size:]]
	Y_test = Y[idx[train_size:]]

	input_shape = (PATCH_INPUT, PATCH_INPUT, 1)

	# ************* NEURAL NET *****************
	model2 = Sequential()

	# Convolution layer with rectified linear activation
	model2.add(Convolution2D(64, 3,3, border_mode='same',
							input_shape=input_shape))
	model2.add(Activation('relu'))

	#model.add(MaxPooling2D(pool_size=(1, 1)))
	model2.add(Flatten())
	model2.add(Dense(1024))
	model2.add(Activation('relu'))
	#model2.add(Dense(output_dim=(PATCH_INPUT*PATCH_INPUT, nb_class) , activation='softmax'))
	model2.add(Dense(PATCH_INPUT*PATCH_INPUT))
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

	return