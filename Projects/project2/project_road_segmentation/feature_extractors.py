import numpy as np

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
