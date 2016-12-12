from PIL import Image, ImageOps
import numpy as np
import math

"""
Notation Cheatsheet:
(ORIGINAL) - function provided by TAs and used in unmodified form
(MODIFIED) - function provided by TAs but modified for our purpose
(CUSTOM) - function created by us
"""

def img_float_to_uint8(img):
    """(ORIGINAL) Image value converter from float to uint with values from 0 to 255

    Args:
        img (3D array): Image as 3D array

    Returns:
        [[[]]]: Image as 3D array (RGB)

    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def rotate_imgs(imgs_array, angle_step, flip=False):
    """(CUSTOM) Image rotating function

    Args:
        imgs_array (array): Array of images (3D numpy arrays).
        angle_step (double): step of rotation angle
        flip (bool): flag. If true, fliped image will also be included.

    Returns:
        [ [[[]]] ]: list of Images as 3D numpy arrays (RGB)

    """
    rotated_imgs = []

    if angle_step:
        number_of_rotations = math.floor(360 / angle_step)
        print('Number of rotations: {0}'.format(number_of_rotations))
        print('Flipping: {0}'.format(flip))
    else:
        print('Skipping rotation.')
        if flip:
            for img in imgs_array:
                is_2d = len(img.shape) < 3
                if is_2d:
                    pil_img = Image.fromarray(img, 'L')
                else:
                    pil_img = Image.fromarray(img, 'RGB')
                flipped = ImageOps.mirror(pil_img)
                rotated_imgs.append(np.array(flipped))
        else:
            print('Skipping flip.')
        return rotated_imgs

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

def img_crop(im, w, h):
    """(ORIGINAL) Crops image into n patches of h x w size

    Args:
        im (3D array): Image as 3D array
        w (int): width of patch
        h (int): height of patch

    Returns:
        [ [[[]]] ]:  List of patches as 3D array (RGB)

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

def patch_range(w, h, w_translation, h_translation):
    """(CUSTOM) Calculate valid patch range for translation of patches

    Args:
        w (int): width of patch
        h (int): height of patch
        w_translation (int): w translation of patch
        h_translation (int): h translaion of patch

    Returns:
        ([], []):  2-tuple of lists containting pixel translations
    """
    h_range = [0] if h_translation <= 0 else list(range(0, h, h_translation))
    minus_h_range = [ x * -1 for x in h_range]
    h_range = list(set(h_range + minus_h_range)) # trick to remove duplicates
    h_range.sort()

    w_range = [0] if w_translation <= 0 else list(range(0, h, h_translation))
    minus_w_range = [ x * -1 for x in w_range]
    w_range = list(set(w_range + minus_w_range)) # trick to remove duplicates
    w_range.sort()

    return h_range, w_range

def translate_img(img, h_range, w_range):
    """(CUSTOM) Calculate all translated images considering provided ranges.

    Args:
        img (2D/3D numpy array): image
        h_range ([int]): list of pixel translations in h direction
        w_range ([int]): list of pixel translations in w direction
    Returns:
        [PIL image]: List containting all translated images as PIL Images
    """
    translations = []
    imgwidth = img.shape[0]
    imgheight = img.shape[1]
    is_2d = len(img.shape) < 3
    if is_2d:
        pil_img = Image.fromarray(img, 'L')
    else:
        pil_img = Image.fromarray(img, 'RGB')

    for it in h_range:
        for jt in w_range:
            translated_img = pil_img.crop((jt, it, jt + imgwidth, it + imgheight))
            translations.append(translated_img)
    return translations

def img_crop(img, w, h, w_translation, h_translation):
    """(MODIFIED) Image cropping functon with optional crop tranlation

    Args:
        img (2D/3D numpy array): image
        w (int): w size of patch
        h (int): h size of patch
        w_translation (int): crop translation in number of pixels for w direction
        h_translation (int): crop translation in number of pixels for h direction
    Returns:
        [ 2D/3D numpy arary]: List containting all patches as Numpy 2D/3D arrays
    """
    list_patches = []
    imgwidth = img.shape[0]
    imgheight = img.shape[1]

    h_range, w_range = patch_range(w, h, w_translation, h_translation)
    transated_imgs = translate_img(img, h_range, w_range)
    for img in transated_imgs:
            for i in range(0,imgheight,h):
                for j in range(0,imgwidth,w):
                    im_patch = img.crop((j, i, j + w, i + h))
                    list_patches.append(np.array(im_patch))
    return list_patches
