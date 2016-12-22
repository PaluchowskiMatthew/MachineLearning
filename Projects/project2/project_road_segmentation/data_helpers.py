import os
import matplotlib.image as mpimg
import numpy as np

from keras.models import load_model

def load_img_gt(data_range):
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

def value_to_class(v, threshold=0.25):
	"""(ORIGINAL) Extract image/patch value to binary class of road(1)/non-road(0)
	Args:
	img (numpy array): Original image/patch
		threshold (double): classifing threshold
	Returns:
		int: class label
	"""
	df = np.sum(v)
	if df > threshold:
		return 1
	else:
		return 0

def value_to_2d_class(v, threshold=0.25):
	"""(MODIFIED) Extract image/patch value to 2d class of road(1)/non-road(0)
		Args:
		img (numpy array): Original image/patch
			threshold (double): classifing threshold
		Returns:
			int: class label
	"""
	mean = np.mean(v)
	df = np.sum(mean)
	if df > threshold:
		return [1, 0] #	***** category matrix
	else:
		return [0, 1]

def extract_data_simple(imgs, gts, patch_size, categorical=True):
	img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(imgs.shape[0])]
	data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

	gt_patches = [img_crop(gts[i], patch_size, patch_size) for i in range(gts.shape[0])]
	gt_data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

	if categorical:
		labels = numpy.asarray([value_to_2d_class(numpy.mean(gt_data[i])) for i in range(len(data))])
		return numpy.asarray(data), numpy.asarray(labels)
	else:
		labels = numpy.asarray([value_to_class(numpy.mean(gt_data[i])) for i in range(len(data))])
		return numpy.asarray(data), labels.astype(numpy.float32)

def extract_data_window(imgs, gts, patch_size, patch_window, image_size, model=False, output_size_model=50):
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
