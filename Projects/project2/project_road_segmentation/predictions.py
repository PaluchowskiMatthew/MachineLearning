from keras.models import load_model
from mask_to_submission import *
from image_handling import *
import numpy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from new_process import new_process

IMG_PATCH_SIZE = 8

"""
TO DO:

PROCESS IMG PREDICTIONS

"""

def predict(model_name):
	data_dir = 'test_set_images/'
	pred_dir = 'predictions/'

	print("launch post")
	new_process(model_name, [1, 10])
	return

	model_path = 'models/' + model_name

	model = load_model(model_path)
	model.compile(loss='categorical_crossentropy',
				   optimizer='adadelta',
				   metrics=['fmeasure'])

	

	"""img = mpimg.imread('training/images/satImage_017.png')
	data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
	predictions_patch = model.predict_classes(data, verbose=1)

	file = open('dummy.txt', 'wb')
	np.savetxt(file, predictions_patch, fmt='%.0d')
	file.close()
	print('done')
	return"""
	
	for i in range(1, 51):
		imageid = "test_%.1d" % i
		image_filename = data_dir + imageid + ".png"
		if os.path.isfile(image_filename):
			print ('Predicting' + image_filename)
			img = mpimg.imread(image_filename)

			data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))

			predictions_patch = model.predict_classes(data, verbose=1)

			img_prediction = label_to_img(img.shape[0], img.shape[1], 
										  IMG_PATCH_SIZE, IMG_PATCH_SIZE, 
										  predictions_patch)

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