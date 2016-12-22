"""
	********* PCML: MINIPROJECT 2 ROAD SEGEMENTATION ***********************

	This function predicts the images of the test set and creates a submission file.

	Function predict():
		Inputs:
			-main_model:  	name of the saved main neural network
			-post_model:    name of the saved post processing network

		Outputs:
			-Prediction folder with the images of the predictions of tests images
			-Submission .CSV file for submission on kaggle


	authors: Maateusz Paaluchowski, Marie Drieghe and Lazare Girardin
"""
# ************ IMPORT LIBRARIES ************************************************
from keras.models import load_model
from mask_to_submission import *
from image_handling import *
import numpy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

### IMPORT PRIMARY MODEL AND POST-PROCESSING MODEL ###
MODEL_NAME = "windows_8x8.h5"
POST_MODEL_NAME = "post_windows_8x8.h5"

model = load_model('models/' + MODEL_NAME)
model.compile(loss='categorical_crossentropy',
                   optimizer='adadelta',
                   metrics=['fmeasure'])
model_post = load_model('models/POST/' + POST_MODEL_NAME)
model_post.compile(loss='categorical_crossentropy',
                   optimizer='adadelta',
                   metrics=['fmeasure'])


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
	new_size = int((608)/PATCH_UNIT)
	p=int(PATCH_UNIT/2)
	X = np.zeros((num_images*new_size**2, PATCH_WINDOW, PATCH_WINDOW, 3))
	# Slide the patch window through each image and assign to each patch the center label of groundtrhuth image
	for im in range(num_images):
		im_off = im*new_size**2 #Image offset
		for i,x in enumerate(range(w+p, w+pred_size, PATCH_UNIT)):
			x_off = new_size*i # x-axis offset
			for j, y in enumerate(range(w+p, w+pred_size, PATCH_UNIT)):
				X[im_off+x_off+j, :, :, :] = imgs[im, (x-w):(x+w+1), (y-w):(y+w+1), :] #data: square corresponding to PATcH_WINDOW labels predicted
	return X

def extract_data_sec(train_range, train_data_filename, file_str, PATCH_UNIT, PATCH_WINDOW, img_SIZE):

    # **** LOAD FIRST MODEL *****************


    # **** LOAD IMAGES ***********************
    w = int((PATCH_WINDOW-1)/2)

    num_images = train_range[1]-train_range[0]+1
    pred_size = int(img_SIZE/PATCH_UNIT)
    pad_size = pred_size + 2*w
    imgs = np.zeros((num_images, pad_size, pad_size))

    for j,i in enumerate(range(train_range[0], train_range[1]+1)):
        data = extract_d([i, i], train_data_filename, file_str, 8, 17, 608)
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

def predict_sec():

	# *********** PARAMETERS ********************************************************
	# Patches used by first CNN
	IMG_PATCH_SIZE = 8
	# Window used by second CNN
	PATCH_WINDOW = 21
	# Size of the test images
	IMG_SIZE = 608


	data_dir = 'test_set_images/'
	pred_dir = 'predictions_CNN_post/'

	new_size = int(IMG_SIZE/IMG_PATCH_SIZE)

	for i in range(1, 51):
		imageid = "test_%.1d" % i
		image_filename = data_dir + imageid + ".png"

		if os.path.isfile(image_filename):
			print ('Predicting' + image_filename)
			img = mpimg.imread(image_filename)

			data = extract_data_sec([i, i], data_dir, "test_%.1d", IMG_PATCH_SIZE, PATCH_WINDOW, IMG_SIZE)

			total_img = model_post.predict_classes(data, verbose=1)

			img_prediction = label_to_img(img.shape[0], img.shape[1],
										  IMG_PATCH_SIZE, IMG_PATCH_SIZE,
										  total_img)
			pimg = Image.fromarray((img_prediction*255.0).astype(np.uint8))
			pimg = pimg.transpose(Image.FLIP_LEFT_RIGHT)
			pimg = pimg.transpose(Image.ROTATE_90)
			#pimg.save(pred_dir + "prediction_" + str(i) + ".png")
			pimg.save(pred_dir + "prediction_" + str(i) + ".png")
		else:
			print ('File ' + image_filename + ' does not exist')

	print("\n Done predicting")
	submission_filename = pred_dir + 'newTry.csv'
	image_filenames = []
	for i in range(1, 51):
		image_filename = pred_dir + 'prediction_' + str(i) + '.png'
		print (image_filename)
		image_filenames.append(image_filename)
		masks_to_submission(submission_filename, *image_filenames)
	print("Submission file finished")

	return
