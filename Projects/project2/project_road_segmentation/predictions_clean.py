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
from submission_helper import *
from data_helpers import *
import matplotlib.image as mpimg
from PIL import Image

### IMPORT PRIMARY MODEL AND POST-PROCESSING MODEL ###
MODEL_NAME = "windows_8x8.h5"
POST_MODEL_NAME = "weights-improvement-00-0.93.h5"

model = load_model('models/' + MODEL_NAME)
model.compile(loss='categorical_crossentropy',
                   optimizer='adadelta',
                   metrics=['fmeasure'])
model_post = load_model('models/POST/' + POST_MODEL_NAME)
model_post.compile(loss='categorical_crossentropy',
                   optimizer='adadelta',
                   metrics=['fmeasure'])

def predict():
	# Parameters of first CNN
	MODEL_PATCH_SIZE = 8
	MODEL_WINDOW_SIZE = 17
	MODEL_IMAGE_SIZE = 608

	# Parameters of post CNN
	POST_PATCH_SIZE = 8
	POST_WINDOW_SIZE = 21
	POST_IMAGE_SIZE = 608

	DATA_DIR = 'test_set_images/'
	PRED_DIR = 'final_predictions/'

	if not os.path.exists(PRED_DIR):
		os.makedirs(PRED_DIR)

	N_TEST_IMG = 10

	pred_size = int(MODEL_IMAGE_SIZE/MODEL_PATCH_SIZE)

	for i in range(1,N_TEST_IMG):
		imageid = "test_%.1d" % i
		image_filename = DATA_DIR + imageid + ".png"

		if os.path.isfile(image_filename):
			print ('Predicting ' + image_filename)
			img = mpimg.imread(image_filename)
			img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
			gt = np.zeros(img.shape)

			data, y = extract_data_model(model, MODEL_PATCH_SIZE, MODEL_WINDOW_SIZE, MODEL_IMAGE_SIZE, img, gt, POST_PATCH_SIZE, POST_WINDOW_SIZE, POST_IMAGE_SIZE)

			total_img = model_post.predict_classes(data, verbose=1)

			img_prediction = label_to_img(img.shape[1], img.shape[2],
										  POST_PATCH_SIZE, POST_PATCH_SIZE,
										  total_img)

			pimg = Image.fromarray((img_prediction*255.0).astype(np.uint8))
			pimg = pimg.transpose(Image.FLIP_LEFT_RIGHT)
			pimg = pimg.transpose(Image.ROTATE_90)

			pimg.save(PRED_DIR + "prediction_" + str(i) + ".png")
		else:
			print ('File ' + image_filename + ' does not exist')

	print("\n Done predicting")

	submission_filename = PRED_DIR + 'new_submission.csv'
	image_filenames = []
	for i in range(1, N_TEST_IMG):
		image_filename = PRED_DIR + 'prediction_' + str(i) + '.png'
		image_filenames.append(image_filename)
		masks_to_submission(submission_filename, *image_filenames)
	print("Submission file finished")

	return
