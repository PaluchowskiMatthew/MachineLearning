import os
import numpy as np
import matplotlib.image as mpimg
import re
from keras.models import load_model
from PIL import Image, ImageOps

from helpers.dataset_preprocessing import load_image_pil
from helpers.image_modifiers import img_crop_translate, img_crop
from helpers.visualization_helpers import label_to_img
from helpers.feature_extractors import patch_to_label


def predict_imgs(data_dir='test_set_images/', prediction_root_dir='predictions/', patch_size=8, patch_translation=0, **kwargs):
    model = kwargs.get('model')
    full_model_path = kwargs.get('full_model_path')

    if not model and not full_model_path:
        print("**kwarg model= or full_model_path= is required!")
        return

    if not os.path.exists(prediction_root_dir):
        os.makedirs(prediction_root_dir)

    if full_model_path:
        model = load_model(full_model_path)

    model.compile(loss='categorical_crossentropy',
    	           optimizer='adadelta',
    	           metrics=['fmeasure'])

    for i in range(1, 51):
        im_id = str(i).zfill(2)
        imageid = "test_%.1d" % i
        image_filename = data_dir + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Predicting ' + image_filename)
            img = mpimg.imread(image_filename)

            data = np.asarray(img_crop(img, patch_size, patch_size))

            predictions_patch = model.predict_classes(data, verbose=1)

            img_prediction = label_to_img(img.shape[0], img.shape[1],
            							  patch_size, patch_size,
            							  predictions_patch)

            pimg = Image.fromarray((img_prediction*255.0).astype(np.uint8))
            pimg = ImageOps.invert(pimg)
            pred_id = str(i).zfill(3)
            pimg.save(prediction_root_dir + "prediction_" + pred_id + ".png")
        else:
            print ('File ' + image_filename + ' does not exist')
    print("Prediction finished.")
    return

def predictions_to_submission(prediction_root_dir='predictions/', submission_filename='cnn_submission.csv', patch_size=8, threshold=0.25):
    image_filenames = []
    for i in range(1, 51):
        pred_id = str(i).zfill(3)
        image_filename = 'prediction_' + pred_id + '.png'
        img_path = prediction_root_dir + image_filename
        print (img_path)
        image_filenames.append(img_path)
        masks_to_submission(submission_filename, patch_size, threshold, *image_filenames)
    print("Submission file finished")


def mask_to_submission_strings(image_filename, patch_size, threshold):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch, threshold)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, patch_size, threshold, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, patch_size, threshold))
