"""
	********* PCML: MINIPROJECT 2 ROAD SEGEMENTATION ***********************

	This function train a second Neural Net used as a generalization of post processing
	morphological functions. The input of the CNN are patches of size PATCH_WINDOW (number of
	patch unit used for the window) and aim to output the center pixel of this window. Labels
	are taken from the groundtruth images.
	The predictions of the first CNN are mirror padded to keep the image of the same size.
	The file post_processing_NN aims the same but without padding, which results in images smaller
	by 2 times the side of the window (called w).

	The CNN is trained using Keras, which is based on TensorFlow (or Theano if chosen).

	Function post_padded():
		Input:
			-model_name:  the name of the first neural net whose predictions are used as inputs
						  The model has to be on path: 'models/'
						  Exemple: 'main_8x8.h5'
			-train range: Range of training images to be trained on (which allows to train the 
						  network on totally different images than the first network). It should be 
						  of the form: [51, 100] for images from 51 to 100.
			-post_name:   Name of the saved trained network, in directory 'models/POST/'
						  Exemple: 'padded_8x8.h5'
		Ouput:
			-trained network in folder 'models/POST', as an .h5 file


	authors: Maateusz Paaluchowski, Marie Drieghe and Lazare Girardin
"""
# **********  IMPORT LIBRARIES  *******************************************************
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, SpatialDropout2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import SGD
from keras import backend as K

from image_handling import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image


#	CALL THIS FUNCTION TO TRAIN SECOND CNN
def post_padd(model_name, train_range, post_name='dummy.h5'):

	# ********** PARAMETERS ***************************************************************
	# Size of the train images
	IMG_SIZE = 400
	# patches used in training of the first CNN
	PATCH_UNIT = 8
	# Number of class to predict
	nb_class = 2
	# Size of the window for which we aim to predict the center
	PATCH_WINDOW = 21  # MUST BE ODD
	# Training batch size
	batch_size = 128
	# Number of epochs for training
	nb_epoch = 7

	#******* EXCTRACT DATA ***********************************************************
	data_dir = 'training/'
	train_data_filename = data_dir + 'images/'
	train_labels_filename = data_dir + 'groundtruth/'
	file_str = "satImage_%.3d"

	pred = extract_data_padded(model_name, train_range, train_data_filename, file_str, 
								PATCH_UNIT, PATCH_WINDOW, IMG_SIZE)
	Y = extract_labels_padded(train_range, train_labels_filename, PATCH_UNIT, PATCH_WINDOW)

	#******** CHECK ACCURACY OF PREDICTIONS OF THE FIRST NN ***************************
	new_size = int(IMG_SIZE/PATCH_UNIT)
	w = int((PATCH_WINDOW-1)/2)
	num_images = train_range[1]-train_range[0]+1
	TP=0.
	FN=0.
	for ii in np.arange((num_images*new_size**2)):
		if pred[ii, w, w] == Y[ii, 0]:
			TP = TP+1.
		else:
			FN = FN +1.
	print("Acc on training images: ")
	print(TP/(TP+FN))


	# ************* CREATE TRAIN AND TEST SETS **************************************
	TRAIN_RATIO = 0.7
	idx = np.random.permutation(np.arange(pred.shape[0]))
	train_size = int(TRAIN_RATIO*pred.shape[0])
	X_train = pred[idx[:train_size]]
	Y_train = Y[idx[:train_size]]
	X_test = pred[idx[train_size:]]
	Y_test = Y[idx[train_size:]]

	# ************* NEURAL NET ******************************************************
	input_shape = (PATCH_WINDOW, PATCH_WINDOW, 1)
	model2 = Sequential()
	# Convolution layer with rectified linear activation
	model2.add(Convolution2D(64, 5,5, border_mode='same',
							input_shape=input_shape))
	model2.add(Activation('relu'))

	#Dropout to avoid overfitting
	model2.add(SpatialDropout2D(0.1))

	# Second convolutional layer
	model2.add(Convolution2D(128, 3,3))
	model2.add(Activation('relu'))

	# Flatten the input to have dense layers
	model2.add(Flatten())
	#model2.add(Dense(1024))
	#model2.add(Activation('relu'))
	#model2.add(Dropout(0.25))

	model2.add(Dense(512))
	model2.add(Activation('relu'))
	model2.add(Dropout(0.25))
	
	model2.add(Dense(nb_class))
	model2.add(Activation('softmax'))

	# Compile the model
	model2.compile(loss='categorical_crossentropy',
			  optimizer='adadelta',
			  metrics=['fmeasure'])

	# Train the model
	model2.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
			  class_weight='auto', verbose=1, validation_data=(X_test, Y_test))

	# Evaluate the model
	score = model2.evaluate(X_test, Y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	# Save the model
	model_path2 = 'models/POST/' + post_name
	model2.save(model_path2)

	# Visualize one image
	"""
	try1 = extract_data_padded(model_name, [3, 3], train_data_filename, file_str,
								PATCH_UNIT, PATCH_WINDOW, IMG_SIZE)
	
	predictions_patch = model2.predict_classes(try1, verbose=1)

	img_prediction = label_to_img(new_size*PATCH_UNIT, new_size*PATCH_UNIT, 
										  PATCH_UNIT, PATCH_UNIT, 
										  predictions_patch)

	pimg = Image.fromarray((img_prediction*255.0).astype(np.uint8))
	plt.imshow(pimg, cmap='Greys_r')
	plt.show()
	"""
	return
