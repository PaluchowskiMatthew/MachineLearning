import matplotlib.image as mpimg
import numpy as np
import skimage as ski
import sklearn as skl
import matplotlib.pyplot as plt
import os,sys
from PIL import Image, ImageOps
import math


def load_image(infilename):
    """Image loding function

    Args:
        infilename (str): Path of image file.

    Returns:
        [[[]]]: Image as 3D array (RGB)

    """
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    """Image value converter from float to uint with values from 0 to 255

    Args:
        img (3D array): Image as 3D array

    Returns:
        [[[]]]: Image as 3D array (RGB)

    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def rotate_imgs(imgs_array, angle_step, flip=False):
    """Image rotating function

    Args:
        imgs_array (array): Array of images (3D numpy arrays).
        angle_step (double): step of rotation angle
        flip (bool): flag. If true fliped image will also be included

    Returns:
        array of 3D numpy arrays[ [[[]]] ]

    """
    number_of_rotations = math.floor(360 / angle_step)
    rotated_imgs = []

    for img in imgs_array:
        for rotation_number in range(1, number_of_rotations):
            is_2d = len(img.shape) < 3
            if is_2d:
                pil_img = Image.fromarray(img, 'L')
            else:
                pil_img = Image.fromarray(img, 'RGB')
            rotated = pil_img.rotate(angle_step * rotation_number)
            rotated_imgs.append(np.array(rotated))
            if flip:
                flipped = ImageOps.mirror(pil_img)
                rotated = flipped.rotate(angle_step * rotation_number)
                rotated_imgs.append(np.array(rotated))
    return rotated_imgs

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

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

# Convert array of labels to an image

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img):
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

# Extract features for a given image
def extract_img_features(filename, patch_size):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))])
    return X

def value_to_class(v, threshold):
    df = np.sum(v)
    if df > threshold:
        return 1
    else:
        return 0
def extract_patches(imgs, patch_size):
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(len(imgs))]
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    return img_patches

def compute_true_positive_rate(groundtruth, prediction):
    # Get non-zeros in prediction and grountruth arrays
    Zn = np.nonzero(prediction)[0]
    Yn = np.nonzero(groundtruth)[0]

    TPR = len(list(set(Yn) & set(Zn))) / float(len(prediction))
    return TPR

def display_prediction(original_image, patch_size, prediction):
    #Display prediction as an image on top of original one
    w = original_image.shape[0]
    h = original_image.shape[1]
    predicted_im = label_to_img(w, h, patch_size, patch_size, prediction)
    cimg = concatenate_images(original_image, predicted_im)
    fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size
    plt.imshow(cimg, cmap='Greys_r')

    new_img = make_img_overlay(original_image, predicted_im)

    plt.imshow(new_img)

def display_prediction_alt(original_image, gt_image, patch_size, prediction):
    #Display prediction as an image on top of original one
    w = original_image.shape[0]
    h = original_image.shape[1]
    predicted_im = label_to_img(w, h, patch_size, patch_size, prediction)

    new_img = make_img_overlay(original_image, predicted_im)

    nChannels = len(gt_image.shape)
    w = gt_image.shape[0]
    h = gt_image.shape[1]
    if nChannels != 3:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_image)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8

        gt_image = gt_img_3c

    new_gt = make_img_overlay(gt_image, predicted_im)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))
    ax1.imshow(original_image)
    ax2.imshow(gt_image)
    ax3.imshow(new_img)
    ax4.imshow(new_gt)
