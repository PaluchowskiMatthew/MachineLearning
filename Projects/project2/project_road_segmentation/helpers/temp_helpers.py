import os, sys
import numpy as np
from helpers.feature_extractors import value_to_class
from helpers.dataset_preprocessing import load_image, img_crop

def extract_img_features_new(img_idx, extract_features, patch_size=16, foreground_threshold=0.25):
    """Build X matrix in the same way as the model data was build for the single image
    Args:
        img_idx (int): ID of the image to extract features from
        extract_features (func): function that extracts features per patch,
        should be the same as used to build the model data
        patch_size (int): Pixel size of patches to divide the images,
        should be the same as used to build the model data
        foreground_threshold (percentage): threshold that decides if a given patch is a road or not,
        should be the same as used to build the model data

    Returns:
        X: matrix of input samples [n_samples, n_features]
    """
    root_dir = "training/"
    image_dir = root_dir + "images/"
    gt_dir = root_dir + "groundtruth/"
    files = os.listdir(image_dir)


    img = load_image(image_dir + files[img_idx])
    gt = load_image(gt_dir + files[img_idx])

    img_patches = img_crop(img, patch_size, patch_size)
    gt_patches = img_crop(gt, patch_size, patch_size)

    X = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))])
    Y = np.asarray([ value_to_class(np.mean(gt_patches[i]), foreground_threshold) for i in range(len(gt_patches))])

    return img, gt, Y, X
