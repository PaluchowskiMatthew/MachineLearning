import matplotlib.image as mpimg
import os,sys
import numpy as np
from image_modifiers import *
from feature_extractors import *

"""
Notation Cheatsheet:
(ORIGINAL) - function provided by TAs and used in unmodified form
(MODIFIED) - function provided by TAs but modified for our purpose
(CUSTOM) - function created by us
"""

def load_image(infilename):
    """(ORIGINAL) Image loding function

    Args:
        infilename (str): Path of image file.

    Returns:
        [[[]]]: Image as 3D array (RGB)

    """
    data = mpimg.imread(infilename)
    return data


def create_dataset(root_dir, dataset_size, train_fraction, **kwargs):
    """(MODIFIED) Dataset loding function

    Args:
        root_dir (str): Path of root directory where dataset is included.
                        WARNING: directory tree needs to be like
                        root_dir:
                            - images (satelite images)
                                - sat1.jpg
                                - ...
                            - groundtruth (binary groundtruth image)
                                - sat1.jpg
                                - ...
                        and images must have the same name in both directories.
        dataset_size (int): Number of pictures to load
        train_fraction (double): Fraction of dataset used for training (rest for test).

    Kwargs - optional arguments:
        rotation_angle (double): Angle step for optional rotation. If 0.0 no rotation will be applied.
        flip (bool): If true then each picture (original and rotated) will also be mirrored.

    Returns:
        (train_imgs, train_gt_imgs, test_imgs, test_gt_imgs): 4 lists of images as 3D numpy Arrays in one 4-tuple

    """
    rotation_angle = kwargs.get('rotation_angle', 0.0)
    flip = kwargs.get('flip', False)

    image_dir = root_dir + 'images/'
    if not os.path.exists(image_dir):
        print('Directory {0} not found!'.format(image_dir))
        return
    included_extenstions = ['jpg', 'bmp', 'png', 'gif', 'tif', 'tiff', 'jpeg']
    files = [fn for fn in os.listdir(image_dir) if any(fn.endswith(ext) for ext in included_extenstions)]
    if len(files) <= 0:
        print('No image files in directory {0} found!'.format(image_dir))
        return
    n = min(dataset_size, len(files))
    print('Original loaded dataset size: {0}'.format(str(n)))
    imgs = [img_float_to_uint8(load_image(image_dir + files[i])) for i in range(n)]

    train_size = int(n*train_fraction)
    train_imgs = imgs[0:train_size]
    test_imgs = imgs[train_size:]
    print('Creating train dataset...')
    imgs_rotated_flipped = rotate_imgs(train_imgs, rotation_angle, flip)
    train_imgs += imgs_rotated_flipped

    gt_dir = root_dir + 'groundtruth/'
    if not os.path.exists(gt_dir):
        print('Directory {0} not found!'.format(gt_dir))
        return
    gt_imgs = [img_float_to_uint8(load_image(gt_dir + files[i])) for i in range(n)]
    train_gt_imgs = gt_imgs[0:train_size]
    test_gt_imgs = gt_imgs[train_size:]
    print('Creating test dataset...')
    gt_imgs_rotated_flipped = rotate_imgs(train_gt_imgs, rotation_angle, flip)
    train_gt_imgs += gt_imgs_rotated_flipped

    print('Created train dataset size: {0}'.format(len(train_imgs)))
    print('Created test dataset size: {0}'.format(len(test_imgs)))

    return train_imgs, test_imgs, train_gt_imgs, test_gt_imgs

def extract_patches(dataset, patch_size, patch_translation):
    """(MODIFIED) Patches extracting function from whole dataset

    Args:
        dataset (4-tuple): 4-Tuple containing lists of images (train, test, train_gt, test_gt).
        patch_size (int): Patch size. Needs to be an increment of 2.
        patch_translation (int): Number of pixels by which patch should be translated.

    Returns:
        [train_patches, test_patches, train_gt_patches, test_gt_patches]: A list of patches of dataset
    """
    names = ['Train patches: ', 'Test patches: ', 'Train GT patches: ', 'Test GT patches: ']
    patches = []
    for i, imgs in enumerate(dataset):
        img_patches = [img_crop_translate(imgs[i], patch_size, patch_size, patch_translation, patch_translation) for i in range(len(imgs))]
        img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
        print(names[i] + '{0}'.format(len(img_patches)))
        patches.append(img_patches)
    return patches

def compute_input_features(input_patches, func, **kwargs):
    """(MODIFIED) Feature computing function for input patches
    Args:
        input_patches ([]): Array of input patches of images
        func (function): Function which should be used to compute fetures
    Returns:
        [train_feature, test_features]: A list of feautures of dataset
    """
    features = []
    names = ['Train features: ', 'Test features: ']
    for i, patch in enumerate(input_patches):
        feature = np.asarray([ func(patch[i]) for i in range(len(patch))])
        print(names[i] + '{0}'.format(len(feature)))
        features.append(feature)
    return features

def compute_output_features(output_patches, func, threshold):
    """(MODIFIED) Feature computing function for input patches
    Args:
        output_patches ([]): Array of output patches of images
        func (function): Function which should be used to compute fetures
        threshold (double): Threshold value for classification
    Returns:
        [train_gt_feature, test_gt_features]: A list of gt feautures of dataset
    """
    features = []
    names = ['Train GT features: ', 'Test GT features: ']
    for i, patch in enumerate(output_patches):
        feature = np.asarray([func(patch[i], threshold) for i in range(len(patch))])
        print(names[i] + '{0}'.format(len(feature)))
        features.append(feature)
    return features
