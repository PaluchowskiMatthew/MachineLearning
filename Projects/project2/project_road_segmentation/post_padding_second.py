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
		for i,x in enumerate(range(w+2, w+pred_size, PATCH_UNIT)):
			x_off = new_size*i # x-axis offset
			for j, y in enumerate(range(w+2, w+pred_size, PATCH_UNIT)):
				X[im_off+x_off+j, :, :, :] = imgs[im, (x-w):(x+w+1), (y-w):(y+w+1), :] #data: square corresponding to PATcH_WINDOW labels predicted
	return X

def extract_data_sec(model_name, train_range, train_data_filename, file_str, PATCH_UNIT, PATCH_WINDOW, img_SIZE):

	# **** LOAD FIRST MODEL *****************
	model_path = 'models/' + model_name

	model = load_model(model_path)
	model.compile(loss='categorical_crossentropy',
				   optimizer='adadelta',
				   metrics=['fmeasure'])

	# **** LOAD IMAGES ***********************
	w = int((PATCH_WINDOW-1)/2)

	num_images = train_range[1]-train_range[0]+1
	pred_size = int(img_SIZE/PATCH_UNIT)
	pad_size = pred_size + 2*w
	imgs = np.zeros((num_images, pad_size, pad_size))

	for j,i in enumerate(range(train_range[0], train_range[1]+1)):
		data = extract_d([i, i], train_data_filename, file_str, 8, 17, 400)
		predictions_patch = model.predict_classes(data, verbose=1)
		padded = np.reshape(predictions_patch, (pred_size, pred_size))
		padded = np.pad(padded, ((w, w), (w, w)), 'symmetric')
		#import pdb;pdb.set_trace()
		# Store the predictions in tensor with shape [image index, patch_index x, patch index y, value of prediction]
		imgs[j, :, :] = padded

	#***** CREATE TENSOR **********************
	X = np.zeros((num_images*(pred_size**2), PATCH_WINDOW, PATCH_WINDOW))
	# Slide the patch window through each image and assign to each patch the center label of groundtrhuth image
	for im in range(num_images):
		im_off = im*pred_size**2 #Image offset
		for x in range(w,pred_size+w):
			x_off = pred_size*(x-w) # x-axis offset
			for y in range(w, pred_size+w):
				y_off = (y-w) # y-axis offset
				X[im_off+x_off+y_off, :, :] = imgs[im, (x-w):(x+w+1), (y-w):(y+w+1)] #data: square corresponding to PATcH_WINDOW labels predicted
	return np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))

def extract_labels_sec(train_range, train_labels_filename, PATCH_UNIT, PATCH_WINDOW):
	nb_class = 2
	num_images = train_range[1]-train_range[0]+1
	pred_size = int(IMG_SIZE/PATCH_UNIT)
	labels = np.zeros((num_images, pred_size, pred_size, nb_class))

	# **** EXTRACT GROUNDTRUTH *****************
	for j, i in enumerate(range(train_range[0], train_range[1]+1)):
		imageid = "satImage_%.3d" % i
		image_filename = train_labels_filename + imageid + ".png"
		if os.path.isfile(image_filename):
			print ('Loading ' + image_filename)
			img = mpimg.imread(image_filename)
			img_patch = img_crop(img, PATCH_UNIT, PATCH_UNIT)
			img_lab = np.asarray([value_to_class(np.mean(np.asarray(img_patch[ii]))) for ii in range(len(img_patch))])
			# Store labels in tensor with shape [image index, patch x, patch y , mean label of patch]
			labels[j, :, :, :] = np.reshape(img_lab, (pred_size, pred_size, nb_class), order='F')
		else:
			print ('File ' + image_filename + ' does not exist')

	# ******** CREATE TENSOR ******************
	Y = np.zeros((num_images*(pred_size**2), nb_class))
	for im in range(num_images):
		im_off = im*(pred_size**2) #Image offset
		for x in range(pred_size):
			x_off = pred_size*x # x-axis offset
			for y in range(pred_size):
				Y[im_off+x_off+y, :] = labels[im, x, y, :] #labels: center pixel of groundtruth image
	return Y

#	CALL THIS FUNCTION TO TRAIN SECOND CNN
def post_padd_sec(model_name, train_range, post_name='dummy.h5'):

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

	# Extract the data from images by predicting with first CNN and padd the results
	pred = extract_data_sec(model_name, train_range, train_data_filename, file_str, 
								PATCH_UNIT, PATCH_WINDOW, IMG_SIZE)
	Y = extract_labels_sec(train_range, train_labels_filename, PATCH_UNIT, PATCH_WINDOW)

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

	"""
	# Train the model
	model2.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
			  class_weight='auto', verbose=1, validation_data=(X_test, Y_test))
	"""

	datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=True,  # randomly flip images
		vertical_flip=False)  # randomly flip images

	# Compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied).
	datagen.fit(X_train)

	# Fit the model on the batches generated by datagen.flow().
	model.fit_generator(datagen.flow(X_train, Y_train,
						batch_size=batch_size),
						samples_per_epoch=X_train.shape[0],
						nb_epoch=nb_epoch,
						lass_weight='auto',
						verbose=1,
						validation_data=(X_test, Y_test))

	
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
