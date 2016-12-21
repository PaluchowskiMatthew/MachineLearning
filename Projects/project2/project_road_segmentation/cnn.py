"""
	********* PCML: MINIPROJECT 2 ROAD SEGEMENTATION ***********************

	This function trains a convolutional neural network over some images. The input
	of the neural network are cropped patches of the images of size IMG_PATCH_SIZE.
	The labels used for training are the mean of patches equivalent to the input that
	were taken from the groundtruth files.

	The CNN is trained using Keras, which is based on TensorFlow (or Theano if chosen).

	Function train_cnn():
		Input:
			-model_name: the name desired to store the resulting trained network
						 Exemple: 'main_8x8.h5'
		Ouput:
			-trained network in folder 'models/', as an .h5 file


	authors: Maateusz Paaluchowski, Marie Drieghe and Lazare Girardin
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import SGD
from keras import backend as K

from image_handling import *


def extract_d(train_range, train_data_filename, file_str, PATCH_UNIT, PATCH_WINDOW, img_SIZE):
 
	 # **** LOAD IMAGES ***********************
	w = int((PATCH_WINDOW-1)/2)

	num_images = train_range[1]-train_range[0]+1
	pred_size = img_SIZE
	pad_size = pred_size + 2*w
	imgs = np.zeros((num_images, pad_size, pad_size, 3))

	for j,i in enumerate(range(train_range[0], train_range[1]+1)):
		imageid = file_str % i
		image_filename = train_data_filename + imageid + ".png"
		if os.path.isfile(image_filename):

			img = mpimg.imread(image_filename)
			padded = np.pad(img, ((w, w), (w, w), (0, 0)), 'symmetric')

			# Store the predictions in tensor with shape [image index, patch_index x, patch index y, value of prediction]
			imgs[j, :, :, :] = padded
		else:
			print ('File ' + image_filename + ' does not exist')

	#***** CREATE TENSOR **********************
	new_size = int((400)/PATCH_UNIT)
	X = np.zeros((num_images*new_size**2, PATCH_WINDOW, PATCH_WINDOW, 3))
	# Slide the patch window through each image and assign to each patch the center label of groundtrhuth image
	for im in range(num_images):
		im_off = im*new_size**2 #Image offset
		for i,x in enumerate(range(w+4, w+pred_size, PATCH_UNIT)):
			x_off = new_size*i # x-axis offset
			for j, y in enumerate(range(w+4, w+pred_size, PATCH_UNIT)):
				X[im_off+x_off+j, :, :, :] = imgs[im, (x-w):(x+w+1), (y-w):(y+w+1), :] #data: square corresponding to PATcH_WINDOW labels predicted
	return X

def extract_l(train_range, train_labels_filename, PATCH_UNIT, PATCH_WINDOW):
	nb_class = 2
	num_images = train_range[1]-train_range[0]+1
	pred_size = int(400/PATCH_UNIT)
	labels = np.zeros((num_images, pred_size, pred_size, nb_class))

	# **** EXTRACT GROUNDTRUTH *****************
	for j, i in enumerate(range(train_range[0], train_range[1]+1)):
		imageid = "satImage_%.3d" % i
		image_filename = train_labels_filename + imageid + ".png"
		if os.path.isfile(image_filename):
			print ('Loading ' + image_filename)
			img = mpimg.imread(image_filename)
			img_patch = img_crop(img, PATCH_UNIT,PATCH_UNIT)
			img_lab = np.asarray([value_to_class(np.mean(np.asarray(img_patch[ii]))) for ii in range(len(img_patch))])
			# Store labels in tensor with shape [image index, patch x, patch y , mean label of patch]
			labels[j, :, :, :] = np.reshape(img_lab, (pred_size, pred_size, nb_class), order='F')
		else:
			print ('File ' + image_filename + ' does not exist')

	Y = np.zeros((num_images*(pred_size**2), nb_class))
	for im in range(num_images):
		im_off = im*(pred_size**2) #Image offset
		for x in range(pred_size):
			x_off = pred_size*x # x-axis offset
			for y in range(pred_size):
				Y[im_off+x_off+y, :] = labels[im, x, y, :] #labels: center pixel of groundtruth image
	return Y

# CALL THIS FUNCTION TO TRAIN FIRST NEURAL NET
def train_cnn(model_name='dummy.h5'):

	# ********** PARAMETERS **********************************************
	# Number of images to used for training this neural network
	NUMBER_IMGS = 50
	# Ratio of patches in the train and test set
	TRAIN_RATIO = 0.8
	# RGB has 3 channels
	NUM_CHANNELS = 3
	# Training batch size
	batch_size = 128
	# Output classes (road and rest)
	nb_classes = 2

	# ********** Tuning parameters: (See Network architecture as well)
	# size of patch of an image to be used as input and output of the neural net
	IMG_PATCH_SIZE = 8
	IMG_WINDOW = 17
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

	input_shape = (IMG_WINDOW, IMG_WINDOW, NUM_CHANNELS)


	# ***************** HANDLE THE DATA **********************************
	data_dir = 'training/'
	train_data_filename = data_dir + 'images/'
	train_labels_filename = data_dir + 'groundtruth/' 

	# Extract data into numpy arrays.
	#data = extract_data(train_data_filename, NUMBER_IMGS, IMG_PATCH_SIZE)
	#labels = extract_labels(train_labels_filename, NUMBER_IMGS, IMG_PATCH_SIZE)
	data = extract_d([1,50], train_data_filename, "satImage_%.3d", IMG_PATCH_SIZE, IMG_WINDOW, 400)
	labels = extract_l([1, 50], train_labels_filename, IMG_PATCH_SIZE, IMG_WINDOW)

	print(data.shape)
	print(labels.shape)

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

	# Full-connected layers
	model.add(Flatten())

	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Dense(512))
	model.add(Activation('relu'))

	# Dropout to avoid overfitting
	model.add(Dropout(0.5))

	#Fully-connected layer to ouptut the resulting class
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	# Compile the model before training
	model.compile(loss='binary_crossentropy',
			  optimizer='adadelta',
			  metrics=['fmeasure'])

	# Another optimzer could be used such as SGD (yield slightly worse results)
	#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	#model.compile(loss='binary_crossentropy',
	#			  optimizer=sgd,
	#			  metrics=['fmeasure'])

	# Train the model
	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
			  class_weight='auto', verbose=1, validation_data=(X_test, Y_test))

	# Evaluate the model on the test set (excluded from training)
	score = model.evaluate(X_test, Y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	# Save the model
	model.save('models/' + model_name)

	return