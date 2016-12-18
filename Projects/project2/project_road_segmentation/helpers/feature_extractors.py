import numpy as np
from skimage import feature, color

from helpers.image_modifiers import img_crop_translate
from IPython.core.debugger import Tracer

"""
Notation Cheatsheet:
(ORIGINAL) - function provided by TAs and used in unmodified form
(MODIFIED) - function provided by TAs but modified for our purpose
(CUSTOM) - function created by us
"""

def extract_features(img):
    """(ORIGINAL) Extract 6-dimensional features consisting of average RGB color as well as variance

    Args:
        img (numpy array): Original image/patch

    Returns:
        [[[]]]: image/patch enhanced with features
    """
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

def extract_features_2d(img):
    """(ORIGINAL) Extract 2-dimensional features consisting of average gray color as well as variance

    Args:
        img (numpy array): Original image/patch

    Returns:
        [[[]]]: image/patch enhanced with features
    """
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

def extract_features_edge(img):
     """(CUSTOM) Extract 7-dimensional features consisting of average RGB color as well as variance + canny edge detector

     Args:
         img (numpy array): Original image/patch

     Returns:
         [[[]]]: image/patch enhanced with features
     """
     feat_m = np.mean(img, axis=(0,1))
     feat_v = np.var(img, axis=(0,1))
     feat_e = np.asarray(feature.canny(color.rgb2gray(img), sigma=5, low_threshold=0, high_threshold=0.15).sum()).reshape(1,)
     feat = np.append(feat_m, feat_v)
     feat = np.hstack([feat, feat_e])
     return feat

def extract_features_cogrey(img):
    feat_e = extract_features_edge(img)

    cov = feature.greycomatrix(color.rgb2gray(img), [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])

    con = feature.greycoprops(cov, 'contrast')[0]
    d = feature.greycoprops(cov, 'dissimilarity')[0]
    h = feature.greycoprops(cov, 'homogeneity')[0]
    e = feature.greycoprops(cov, 'energy')[0]
    cor = feature.greycoprops(cov, 'correlation')[0]
    a = feature.greycoprops(cov, 'ASM')[0]

    feat = np.hstack([feat_e, con, cor, d, h, e, a])

    return feat



def value_to_class(v, threshold):
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

def value_to_2d_class(v, threshold):
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
        return [1, 0] #              *****  category matrix
    else:
        return [0, 1]

def img_to_label(img, w, h, func, threshold):
    """(ORIGINAL) Convert array of labels to an image
    Args:
        img ([[[]]]): Image as 3D array (RGB)
        w (int): width of patch
        h (int): height of patch
        funct (function): extraction function (value_to_class or value_to_2d_class)

    Returns:
        labels ([]): array of labels for patches
    """
    img_patches = img_crop_translate(img, w, h, 0, 0)
    #img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    labels = np.asarray([func(img_patches[i], threshold) for i in range(len(img_patches))])

    return labels

def patch_to_label(patch, threshold):
    """(MODIFIED) assign a label to a patch
    Args:
        patch ([[[]]]): Image patch as 3D array (RGB)
        threshold (float): classification threshold
    Returns:
        label (int): label for patch
    """
    df = np.mean(patch)
    if df > threshold:
        return 1
    else:
        return 0
