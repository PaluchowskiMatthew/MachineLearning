import os
import matplotlib.image as mpimg
import numpy as np

from keras.models import load_model

"""
Notation Cheatsheet:
(ORIGINAL) - function provided by TAs and used in unmodified form
(MODIFIED) - function provided by TAs but modified for our purpose
(CUSTOM) - function created by us
"""

def load_img_gt(data_range):
	""" (CUSTOM)
	Function loading input images alongside ground truth images.
	WARNING: directory tree needs to be like
	training:
		- images (satelite images)
			- satImage_1.jpg
			- ...
		- groundtruth (binary groundtruth image)
			- satImage_1.jpg
			- ...
	and images must have the same name in both directories.

	Args:
		- data_range ([]): range of images to load as a list containing beginning and end img
			Example: [1,100]

    Returns:
        imgs, gts (([],[])): 2-tuple of lists containiing images and ground truths
    """
	DATA_DIR = 'training/'
	IMAGES_FOLDER = DATA_DIR + 'images/'
	GROUNDTRUTH_FOLDER = DATA_DIR + 'groundtruth/'
	FILENAME_STR = 'satImage_%.3d'

	imgs = []
	gts = []

	for i in range(data_range[0], data_range[1]+1):
		imageid = FILENAME_STR % i
		image_path = IMAGES_FOLDER + imageid + ".png"
		groundtruth_path = GROUNDTRUTH_FOLDER + imageid + ".png"
		if os.path.isfile(image_path):
			img = mpimg.imread(image_path)
			imgs.append(img)
			print ('Loading file ' + image_path)
		else:
			print ('File ' + image_path + ' does not exist')

		if os.path.isfile(groundtruth_path):
			gt = mpimg.imread(groundtruth_path)
			gts.append(gt)
			print ('Loading file ' + groundtruth_path)
		else:
			print ('File ' + groundtruth_path + ' does not exist')

	return imgs, gts

def img_crop(im, w, h):
	"""(ORIGINAL) Crops image into n patches of h x w size

    Args:
        im (3D array): Image as a 3D array
        w (int): width of a patch
        h (int): height of a patch

    Returns:
        [ [[[]]] ]:  List of patches as 3D/2D array (binary/RGB)

    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def value_to_class(img, threshold=0.25):
	"""(ORIGINAL) Extract image/patch value to binary class of road(1)/non-road(0)
	Args:
		img (numpy array): Original image/patch
		threshold (double): classifing threshold
	Returns:
		int: class label
	"""
	df = np.sum(img)
	if df > threshold:
		return 1
	else:
		return 0

def value_to_2d_class(img, threshold=0.25):
	"""(MODIFIED) Extract image/patch value to 2d class of road(1)/non-road(0)
		Args:
			img (numpy array): Original image/patch
			threshold (double): classifing threshold
		Returns:
			[int, int]: class label in matrix form
	"""
	mean = np.mean(img)
	df = np.sum(mean)
	if df > threshold:
		return [1, 0] #	***** category matrix
	else:
		return [0, 1]

def extract_data_simple(imgs, gt_imgs, patch_size, categorical=True):
	"""(CUSTOM) Extract image/patch value either to 1D or to 2D class of road(1)/non-road(0)
		Args:
			imgs ([numpy array]): List of input images
			gt_imgs ([numpy array]): List of ground truths
			patch_size (int): size of the square patch in pixels.
			categorical (bool): flag indicating whether if 1D or 2D classification should be used
		Returns:
			int: class label
	"""
	img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(len(imgs))]
	data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

	gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(len(gt_imgs))]
	data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
	if categorical:
		labels = np.asarray([value_to_2d_class(np.mean(data[i])) for i in range(len(data))])
		return np.asarray(data), np.asarray(labels)
	else:
		labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])
		return np.asarray(data), labels.astype(np.float32)

def extract_data_window(imgs, gts, patch_size, patch_window, image_size, model=False, output_size_model=50):
	"""(CUSTOM) TODO: Lazare's explanation
		Args:
			imgs ([numpy array]): List of input images
			gts ([numpy array]): List of ground truths
			patch_size (int): size of the square patch in pixels.
			patch_window (int): size of the slinding window
			image_size (int): size of the single input image
			model (bool): flag indicating whether input images are output from the previous model (binary images) or not
			output_size_model (int): goes along the model flag - if model==True, output_size_model= number of imgs returned by first model.
		Returns:
			data, Y: extracted patches and lables in form accepted by Keras model.
	"""
	nb_class = 2
	w = int((patch_window-1)/2)
	num_images = len(imgs)
	pred_size_label = int(image_size/patch_size)

	if model:
		p = 0
		step = 1
		channels = 1
		pred_size = output_size_model
	else:
		p = int(patch_size/2)
		step = patch_size
		channels = 3
		pred_size = image_size

	pad_size = pred_size + 2*w

	pad_imgs = np.zeros((num_images, pad_size, pad_size, channels))
	labels = np.zeros((num_images, pred_size_label, pred_size_label, nb_class))

	new_size = int((image_size)/patch_size)
	data = np.zeros((num_images*new_size**2, patch_window, patch_window, channels))
	Y = np.zeros((num_images*(pred_size_label**2), nb_class))

	for im in range(num_images):
		img = imgs[im]
		gt = gts[im]

		padded = np.pad(img, ((w, w), (w, w), (0, 0)), 'symmetric')
		pad_imgs[im, :, :, :] = padded

		im_off = im*new_size**2
		for i,x in enumerate(range(w+p, w+pred_size, step)):
			x_off = new_size*i
			for j, y in enumerate(range(w+p, w+pred_size, step)):
				data[im_off+x_off+j, :, :, :] = pad_imgs[im, (x-w):(x+w+1), (y-w):(y+w+1), :]

		img_patch = img_crop(gt, patch_size, patch_size)
		img_label = np.asarray([value_to_2d_class(np.mean(np.asarray(img_patch[ii]))) for ii in range(len(img_patch))])
		labels[im, :, :, :] = np.reshape(img_label, (pred_size_label, pred_size_label, nb_class), order='F')

		im_off = im*(pred_size_label**2)
		for x in range(pred_size_label):
			x_off = pred_size_label*x
			for y in range(pred_size_label):
				Y[im_off+x_off+y, :] = labels[im, x, y, :]

	return data, Y

def extract_data_model(model_name, patch_size_model, patch_window_model, image_size_model, imgs, gts, patch_size, patch_window, image_size):
	"""(CUSTOM) TODO: Lazare's explanation
		Args:
			model_name (string): name of the model to load with extension.
				Example: 'model.h5'
			patch_size_model (int): patch size for the first model
			patch_window_model (int): size of the sliding window for the first model
			image_size_model (int): size of the single input image for the first model
			imgs ([numpy array]): List of input images
			gts ([numpy array]): List of ground truths
			patch_size (int): size of the square patch in pixels.
			patch_window (int): size of the slinding window
			image_size (int): size of the single input image
		Returns:
			data, Y: extracted patches and lables in form accepted by Keras model.
	"""
	# load model of first cnn
	MODEL_PATH = 'models/' + model_name
	model = load_model(MODEL_PATH)
	model.compile(loss='categorical_crossentropy',
				   optimizer='adadelta',
				   metrics=['fmeasure'])

	return extract_data_model(model, patch_size_model, patch_window_model, image_size_model, imgs, gts, patch_size, patch_window, image_size)


def extract_data_model(model, patch_size_model, patch_window_model, image_size_model, imgs, gts, patch_size, patch_window, image_size):
	output_size_model = int(image_size_model/patch_size_model)

	# extract sliding window input to make predictions with the first model
	data, Y = extract_data_window(imgs, gts, patch_size_model, patch_window_model, image_size_model)
	predictions = model.predict_classes(data, verbose=1)
	pred_imgs = np.reshape(predictions, (len(imgs), output_size_model, output_size_model, 1))

	return extract_data_window(pred_imgs, gts, patch_size, patch_window, image_size, True, output_size_model)
