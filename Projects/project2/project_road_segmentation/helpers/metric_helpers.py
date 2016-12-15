import numpy as np
import math
from helpers.image_modifiers import patch_range

def compute_true_positive_rate(groundtruth, prediction):
    # Get non-zeros in prediction and grountruth arrays
    Zn = np.nonzero(prediction)[0]
    Yn = np.nonzero(groundtruth)[0]

    TPR = len(list(set(Yn) & set(Zn))) / float(len(prediction))
    return TPR

def calculate_train_dataset_size(no_of_imgs, train_fraction, original_img_dimensions, patch_dimensions, patch_translations, angle_step, flip):
    train_imgs = no_of_imgs * train_fraction
    h_range, w_range = patch_range(patch_dimensions[0], patch_dimensions[1], patch_translations[0], patch_translations[1])
    translation_factor = len(h_range) * len(w_range)
    if angle_step:
        number_of_rotations = math.floor(90 / angle_step)
    else:
        number_of_rotations = 0
    if flip:
        flip_factor = 2
    else:
        flip_factor = 1
    size = (train_imgs + train_imgs * number_of_rotations) * flip_factor * translation_factor  * \
    (original_img_dimensions[0]/patch_dimensions[0]) * \
    (original_img_dimensions[1]/patch_dimensions[1])
    return size
