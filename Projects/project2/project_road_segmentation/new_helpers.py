import os, sys
import numpy as np
from original_helpers import load_image, img_crop, value_to_class

def build_model_data(extract_features, n_img=10, patch_size=16, foreground_threshold=0.25):
    """Build data that can be directly used to build a model
    Args:
        n_img (int): Number of images to use to build X.
        patch_size (int): Pixel size of patches to divide the images
        extract_features (func): function that extracts features per patch
        foreground_threshold (percentage): threshold that decides if a given patch is a road or not

    Returns:
        X: matrix of input samples [n_samples, n_features]
        Y: vector of labels (0,1)
    """
    root_dir = "training/"
    image_dir = root_dir + "images/"
    files = os.listdir(image_dir)
    n = min(n_img, len(files)) # Load maximum n_img images
    imgs = [load_image(image_dir + files[i]) for i in range(n)]

    gt_dir = root_dir + "groundtruth/"
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]

    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    X = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))])
    Y = np.asarray([ value_to_class(np.mean(gt_patches[i]), foreground_threshold) for i in range(len(gt_patches))])

    print("X [{}] and Y [{}]".format(X.shape, Y.shape))

    return X, Y

def extract_img_features_new(img_idx, extract_features, patch_size=16, foreground_threshold=0.25):
    """Build X matrix in the same way as the model data was build
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

def pretty_confusion(labels, y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    c = confusion_matrix(y_true, y_pred)

    row_format ="{:>15}" * (c.shape[0] + 1)
    print(row_format.format("t/p", labels[0], labels[1]))
    for i in range(c.shape[0]):
        print(row_format.format(labels[i], c[i][0], c[i][1]))
