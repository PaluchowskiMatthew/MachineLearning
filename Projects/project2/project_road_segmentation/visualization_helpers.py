import numpy as np
from PIL import Image
from image_modifiers import img_float_to_uint8

"""
Notation Cheatsheet:
(ORIGINAL) - function provided by TAs and used in unmodified form
(MODIFIED) - function provided by TAs but modified for our purpose
(CUSTOM) - function created by us
"""

def label_to_img(imgwidth, imgheight, w, h, labels):
    """(ORIGINAL) Convert array of labels to an image
    Args:
        imgwidth (int): width of image
        imgheight (int): height of image
        w (int): width of patch
        h (int): height of patch
        labels ([]): array of labels for patches

    Returns:
        [[[]]]: Image as 3D array (RGB)
    """
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    """(ORIGINAL) Overlay prediction on top of image
    Args:
        img (numpy array): image
        predicted_img (numpy array): predictions as image

    Returns:
        PIL Image with overlay
    """
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def concatenate_images(img, gt_img):
    """(ORIGINAL) Concatenate an image and its groundtruth
    Args:
        img (numpy array): image
        gt_img (numpy array): ground truth image

    Returns:
        PIL Image with ground truth
    """
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

def display_prediction(original_image, patch_size, prediction):
    """(ORIGINAL) Display prediction as an image on top of original one
    Args:
        original_image (numpy array):  original image
        patch_size (int): size of the patch
        prediction ([]): array of labels
    Returns:
        Nothing. Displays prediction.
    """
    w = original_image.shape[0]
    h = original_image.shape[1]
    predicted_im = label_to_img(w, h, patch_size, patch_size, prediction)
    cimg = concatenate_images(original_image, predicted_im)
    fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size
    plt.imshow(cimg, cmap='Greys_r')

    new_img = make_img_overlay(original_image, predicted_im)

    plt.imshow(new_img)

def display_prediction_alt(original_image, gt_image, patch_size, prediction):
    """(MODIFIED) Display prediction as series of images
    Args:
        original_image (numpy array):  original image
        gt_img (numpy array): ground truth image
        patch_size (int): size of the patch
        prediction ([]): array of labels
    Returns:
        Nothing. Displays prediction.
    """
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

def display_feature_statistics(X, Y):
    """(CUSTOM) Display statistics about created features
    Args:
        X ([]):  array of input features
        Y ([]):  array of output features

    Returns:
        Nothing. Displays statistics.
    """
    print('Computed ' + str(X.shape[0]) + ' features')
    print('Feature dimension = ' + str(X.shape[1]))
    print('Number of classes = ' + str(np.max(Y)))

    Y0 = [i for i, j in enumerate(Y) if j == 0]
    Y1 = [i for i, j in enumerate(Y) if j == 1]
    print('Class 0: ' + str(len(Y0)) + ' samples')
    print('Class 1: ' + str(len(Y1)) + ' samples')

def pretty_confusion(labels, y_true, y_pred):
    """(CUSTOM) Display pretty confusion matrix
    Args:
        labels ([]): array of labels
        y_true ([]):  array(?) of true output values
        y_pred ([]):  array(?) of predicted output values

    Returns:
        Nothing. Displays statistics.
    """
    from sklearn.metrics import confusion_matrix

    c = confusion_matrix(y_true, y_pred)

    row_format ="{:>15}" * (c.shape[0] + 1)
    print(row_format.format("t/p", labels[0], labels[1]))
    for i in range(c.shape[0]):
        print(row_format.format(labels[i], c[i][0], c[i][1]))
