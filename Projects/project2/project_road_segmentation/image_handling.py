import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from keras.models import load_model

IMG_SIZE = 400

# Extract patches from a given image
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

def extract_data(filename, num_images, IMG_PATCH_SIZE):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = np.asarray([img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)])
    data = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])

    return np.asarray(data)

def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        return [1, 0] #              *****  category matrix
    else:
        return [0, 1]


def bin_value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        return 0 
    else:
        return 1

def extract_labels(filename, num_images, IMG_PATCH_SIZE):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if labels[idx] > 0.5:
                l = 1
            else:
                l = 0
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels

def extract_data_post(model_name, train_range, train_data_filename, PATCH_UNIT, PATCH_WINDOW):
    
    # **** LOAD FIRST MODEL *****************
    model_path = 'models/' + model_name

    model = load_model(model_path)
    model.compile(loss='categorical_crossentropy',
                   optimizer='adadelta',
                   metrics=['fmeasure'])

    # **** LOAD IMAGES ***********************
    num_images = train_range[1]-train_range[0]+1
    new_size = int(IMG_SIZE/PATCH_UNIT)
    imgs = np.zeros((num_images, new_size, new_size, 1))

    for j,i in enumerate(range(train_range[0], train_range[1]+1)):
        imageid = "satImage_%.3d" % i
        image_filename = train_data_filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Predicting' + image_filename)
            img = mpimg.imread(image_filename)
            data = np.asarray(img_crop(img, PATCH_UNIT, PATCH_UNIT))
            predictions_patch = model.predict_classes(data, verbose=1)

            #import pdb;pdb.set_trace()
            # Store the predictions in tensor with shape [image index, patch_index x, patch index y, value of prediction]
            imgs[j, :, :, :] = np.reshape(predictions_patch, (new_size, new_size, 1), order='C')
        else:
            print ('File ' + image_filename + ' does not exist')

    #***** CREATE TENSOR **********************
    w = int((PATCH_WINDOW-1)/2)
    size_tr = int(new_size - 2*w)
    X = np.zeros((num_images*(size_tr**2), PATCH_WINDOW, PATCH_WINDOW, 1))
    # Slide the patch window through each image and assign to each patch the center label of groundtrhuth image
    for im in range(num_images):
        im_off = im*size_tr**2#im*((size_tr)**2) #Image offset
        for x in range(w,size_tr+w):
            x_off = size_tr*(x-w) # x-axis offset
            for y in range(w, size_tr+w):
                y_off = (y-w) # y-axis offset
                X[im_off+x_off+y_off, :, :] = imgs[im, (x-w):(x+w+1), (y-w):(y+w+1)] #data: square corresponding to PATcH_WINDOW labels predicted
    return X

def extract_labels_post(train_range, train_labels_filename, PATCH_UNIT, PATCH_WINDOW):

    nb_class = 2
    num_images = train_range[1]-train_range[0]+1
    new_size = int(IMG_SIZE/PATCH_UNIT)
    labels = np.zeros((num_images, new_size, new_size, nb_class))

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
            labels[j, :, :, :] = np.reshape(img_lab, (new_size, new_size, nb_class))
        else:
            print ('File ' + image_filename + ' does not exist')

    # ******** CREATE TENSOR ******************
    w = int((PATCH_WINDOW-1)/2)
    size_tr = int(new_size - 2*w)
    Y = np.zeros((num_images*(size_tr**2), nb_class))
    for im in range(num_images):
        im_off = im*size_tr**2#im*((size_tr)**2) #Image offset
        for x in range(w,size_tr+w):
            x_off = size_tr*(x-w) # x-axis offset
            for y in range(w, size_tr+w):
                Y[im_off+x_off+(y-w), :] = labels[im, x, y, :] #labels: center pixel of groundtruth image
    return Y