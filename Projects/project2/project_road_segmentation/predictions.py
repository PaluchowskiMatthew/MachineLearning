from keras.models import load_model
from mask_to_submission import *
from image_handling import *
import numpy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from post_processing_NN import post_process

IMG_PATCH_SIZE = 8
PATCH_WINDOW = 11
IMG_SIZE = 608

def predict(model_name):
	"""
		Creates predictions of the test images.
		Inputs:
			-main_model:  	name of the saved main neural network
			-post_model:    name of the saved post processing network

		Outputs:
			-Prediction folder with the images of the predictions of tests images
			-Submission .CSV file for submission on kaggle
	"""
	data_dir = 'test_set_images/'
	pred_dir = 'predictions/'

	#print("launch post")
	#post_process(model_name, [71, 100])

	model_path = 'models/' + model_name

	main_model = load_model(model_path)
	main_model.compile(loss='categorical_crossentropy',
				   optimizer='adadelta',
				   metrics=['fmeasure'])

	model_path2 = 'models/POST/post.h5'

	model_post = load_model(model_path2)
	model_post.compile(loss='categorical_crossentropy',
				   optimizer='adadelta',
				   metrics=['fmeasure'])

	new_size = int(IMG_SIZE/IMG_PATCH_SIZE)
	w = int((PATCH_WINDOW-1)/2)
	size_tr = int(new_size - 2*w) 

	for i in range(1, 51):
		imageid = "test_%.1d" % i
		image_filename = data_dir + imageid + ".png"
		if os.path.isfile(image_filename):
			print ('Predicting' + image_filename)
			img = mpimg.imread(image_filename)

			data_cnn_1 = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))

			predictions_patch_cnn_1 = main_model.predict_classes(data_cnn_1, verbose=1)

			data_cnn_2 = numpy.zeros((size_tr**2, PATCH_WINDOW, PATCH_WINDOW,1))
			for x in range(w,size_tr+w):
				x_off = size_tr*(x-w) # x-axis offset
				for y in range(w, size_tr+w):
					y_off = (y-w) # y-axis offset
					data_cnn_2[x_off+y_off, :, :] = numpy.reshape(predictions_patch_cnn_1, (new_size, new_size, 1))[(x-w):(x+w+1), (y-w):(y+w+1)]

			predictions_patch_cnn_2 = model_post.predict_classes(data_cnn_2, verbose=1)

			total_img = numpy.reshape(predictions_patch_cnn_1, (new_size, new_size))
			center_img = numpy.reshape(predictions_patch_cnn_2, (size_tr, size_tr))
			for x in range(w, size_tr+w):
				for y in range(w, size_tr+w):
					total_img[x, y] = -(center_img[x-w, y-w]-1)

			img_prediction = label_to_img(img.shape[0], img.shape[1], 
										  IMG_PATCH_SIZE, IMG_PATCH_SIZE, 
										  numpy.reshape(total_img, (new_size*new_size)))

			pimg = Image.fromarray((img_prediction*255.0).astype(np.uint8))
			pimg.save(pred_dir + "prediction_" + str(i) + ".png")
		else:
			print ('File ' + image_filename + ' does not exist')


	print("\n Done predicting")
	submission_filename = 'cnn_try1.csv'
	image_filenames = []
	for i in range(1, 51):
		image_filename = 'predictions/prediction_' + str(i) + '.png'
		print (image_filename)
		image_filenames.append(image_filename)
		masks_to_submission(submission_filename, *image_filenames)
	print("Submission file finished")

	return