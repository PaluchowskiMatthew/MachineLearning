import numpy as np
from PIL import Image, ImageOps, ImageChops
from image_modifiers import patch_range

"""
Notation Cheatsheet:
(ORIGINAL) - function provided by TAs and used in unmodified form
(MODIFIED) - function provided by TAs but modified for our purpose
(CUSTOM) - function created by us
"""

def unpatch(patches_array, patch_size, patch_translation, original_img_width, originl_img_height):
    """(CUSTOM) Unpatching function used when patch translation was used.
                Majority vote for label of each pixel is conducted.

    Args:
        patches_array ([]): Array of patches of images
        patch_size (int): patch size in pixels (asumes patch is square)
        patch_translation (int): patch translation in pixels
        original_img_width (int): width of original image which was patched
        original_img_height (int): height of original image which was patched

    Returns:
        [ [[[]]] ]: list of unpatched images as numpy arrays

    """
    h_patches, w_patches = patch_range(patch_size, patch_size, patch_translation, patch_translation)
    h_patches_rescaled = [x + abs(min(h_patches)) for x in h_patches]
    w_patches_rescaled = [x + abs(min(w_patches)) for x in w_patches]
    number_of_patches = (len(h_patches) * len(w_patches))

    number_of_pictures = patches_array.shape[0] / number_of_patches
    if not number_of_pictures.is_integer() :
        print('Something is wrong. Dimensions are incorrect!')
        return

    unpatched_imgs = []
    for img_no in range(int(number_of_pictures)):
        pil_patched_image = Image.new('L', (original_img_width + w_patches_rescaled[-1], originl_img_height + h_patches_rescaled[-1]))
        pil_index_image = pil_patched_image.copy()

        for j, h_patch in enumerate(h_patches_rescaled):
            for i, w_patch in enumerate(w_patches_rescaled):
                patch_mask = Image.new('L', (original_img_width + w_patches_rescaled[-1], originl_img_height + h_patches_rescaled[-1]))
                index_mask = patch_mask.copy()

                prediction_index = (img_no * number_of_patches) + j * len(h_patches_rescaled) + i
                patch = np.ones((patch_size, patch_size), dtype='uint8')

                prediction_patch = patch * patches_array[prediction_index]
                index_patch = patch

                pil_prediction_patch = Image.fromarray(prediction_patch, 'L')
                pil_index_patch = Image.fromarray(index_patch, 'L')

                upper_left_corner = (w_patch, h_patch)
                patch_mask.paste(pil_prediction_patch, upper_left_corner)
                index_mask.paste(pil_index_patch, upper_left_corner)

                pil_patched_image = ImageChops.add(pil_patched_image, patch_mask)
                pil_index_image = ImageChops.add(pil_index_image, index_mask)

        patched_image = np.array(pil_patched_image)
        index_image = np.array(pil_index_image)

        h_from = h_patches[-1]
        h_to = (originl_img_height + h_patches[-1])
        w_from = w_patches[-1]
        w_to = (original_img_width + w_patches[-1])
        unpatched_img = np.around((patched_image/index_image))[h_from:h_to, w_from:w_to]
        unpatched_imgs.append(unpatched_img)
    return unpatched_imgs
