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
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

IMG_SIZE = 400
PATCH_INPUT = 10
PATCH_UNIT = 8
nb_class = 2

PATCH_WINDOW = 11  # MUST BE ODD

batch_size = 50
nb_epoch = 3

def new_process(model_name, train_range):

	model_path = 'models/' + model_name

	model = load_model(model_path)
	model.compile(loss='categorical_crossentropy',
				   optimizer='adadelta',
				   metrics=['fmeasure'])

	data_dir = 'training/'
	train_data_filename = data_dir + 'images/'
	train_labels_filename = data_dir + 'groundtruth/'

	num_images = train_range[1]-train_range[0]+1
	new_size = int(IMG_SIZE/PATCH_UNIT)
	imgs = np.zeros((num_images, new_size, new_size, 1))
	j=0
	#PREDICT THE IMAGES WITH THE FIRST MODEL
	for i in range(train_range[0], train_range[1]+1):
		imageid = "satImage_%.3d" % i
		image_filename = train_data_filename + imageid + ".png"
		if os.path.isfile(image_filename):
			print ('Predicting' + image_filename)
			img = mpimg.imread(image_filename)
			data = np.asarray(img_crop(img, PATCH_UNIT, PATCH_UNIT))
			predictions_patch = model.predict_classes(data, verbose=1)

			"""
			img_prediction = label_to_img(img.shape[0], img.shape[1], 
										  PATCH_UNIT, PATCH_UNIT, 
										  predictions_patch)
			pimg = Image.fromarray((img_prediction*255.0).astype(np.uint8))
			plt.imshow(pimg, cmap='Greys_r')
			plt.show()

			print(img_prediction.shape)
			pimg = Image.fromarray((np.reshape(predictions_patch, (new_size, new_size), order='F')*255.0).astype(np.uint8))
			plt.imshow(pimg, cmap='Greys_r')
			plt.show()
			"""	

			#import pdb;pdb.set_trace()
			# Store the predictions in tensor with shape [image index, patch_index x, patch index y, value of prediction]
			imgs[j, :, :, :] = np.reshape(predictions_patch, (new_size, new_size, 1), order='F')
			j+1
		else:
			print ('File ' + image_filename + ' does not exist')

	#GET GROUNDTRUTH
	j=0
	labels = np.zeros((num_images, new_size, new_size, nb_class))
	ll = np.zeros((num_images, new_size*new_size, nb_class))
	for i in range(train_range[0], train_range[1]+1):
		imageid = "satImage_%.3d" % i
		image_filename = train_labels_filename + imageid + ".png"
		if os.path.isfile(image_filename):
			print ('Loading ' + image_filename)
			img = mpimg.imread(image_filename)
			img_patch = img_crop(img, PATCH_UNIT, PATCH_UNIT)
			img_lab = np.asarray([value_to_class(np.mean(np.asarray(img_patch[ii]))) for ii in range(len(img_patch))])
			# Store labels in tensor with shape [image index, patch x, patch y , mean label of patch]
			labels[j, :, :, :] = np.reshape(img_lab, (new_size, new_size, nb_class))
			ll[j, :, :] = np.asarray(img_lab)
			j = j+1
		else:
			print ('File ' + image_filename + ' does not exist')

	w = int((PATCH_WINDOW-1)/2)
	size_tr = int(new_size - 2*w)
	pred = np.zeros((num_images*(size_tr**2), PATCH_WINDOW, PATCH_WINDOW, 1))
	Y = np.zeros((num_images*(size_tr**2), nb_class))
	
	# Slide the patch window through each image and assign to each patch the center label of groundtrhuth image
	TP=0
	FN=0
	for im in range(num_images):
		im_off = im*(size_tr**2) #Image offset
		for x in range(w,size_tr+w):
			x_off = size_tr*(x-w) # x-axis offset
			for y in range(w, size_tr+w):
				y_off = (y-w) # y-axis offset
				pred[im_off+x_off+y_off, :, :, :] = imgs[im, (x-w):(x+w+1), (y-w):(y+w+1)] #data: square corresponding to PATcH_WINDOW labels predicted
				Y[im_off+x_off+y_off, :] = labels[im, x, y, :] #labels: center pixel of groundtruth image
				if (pred[im_off+x_off+y_off, 5, 5]==Y[im_off+x_off+y_off, 0]):
					TP = TP+1
				else:
					FN = FN+1

	print("Number of true: ")
	print(TP)
	print("Number of false: ")
	print(FN)
	import pdb;pdb.set_trace()

	# ************* TRAIN AND TEST SETS *******
	TRAIN_RATIO = 0.8
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

	model2.add(MaxPooling2D(pool_size=(2, 2)))
	model2.add(Flatten())
	model2.add(Dense(512))
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