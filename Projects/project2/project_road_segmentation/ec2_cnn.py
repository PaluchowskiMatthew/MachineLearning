from helpers.dataset_preprocessing import create_dataset, extract_patches, compute_input_features, compute_output_features
from helpers.dataset_postprocessing import unpatch
from helpers.feature_extractors import extract_features, extract_features_2d, value_to_class, value_to_2d_class
from helpers.visualization_helpers import *
from helpers.metric_helpers import calculate_train_dataset_size

import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import numpy as np


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D, GaussianDropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


def train_cnn(model_name='ec2_model.h5'):
    # GLOBAL VARIABLES
    ROOT_DIR = "training/"
    TOTAL_IMAGES = 100 # Number of images to load
    TRAIN_FRACTION = 0.8 # Percentage of images used for training
    ANGLE_STEP = False # Gotta be 90/ANGLE_STEP needs to be an integer
    FLIP = False # Flag to signal if flipped  versions of rotated images should also be created
    PATCH_SIZE = 8
    PATCH_TRANSLATION = 0 # WARNING: this quickly explodes to enormous amount of data if small patch_translation is selected.
    FOREGROUND_THRESHOLD = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    ORIGINAL_IMAGE_WIDTH = 400
    ORIGINAL_IMAGE_HEIGHT = 400
    NUM_CHANNELS = 3
    THEANO = False

    batch_size = 500
    nb_classes = 2


    # ********** Tuning parameters: (See Network architecture as well)

    # Epochs to be trained
    nb_epoch = 100
    # number of convolutional filters to use
    nb_filters = 64
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (4, 3)

    """
    if "image_dim_ordering": is "th" and "backend": "theano", your input_shape must be (channels, height, width)
    if "image_dim_ordering": is "tf" and "backend": "tensorflow", your input_shape must be (height, width, channels)
    """
    if THEANO:
        input_shape = (NUM_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    else:
        input_shape = (PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS)

    ############### DATA SET CREATON

    train_dataset_size = calculate_train_dataset_size(TOTAL_IMAGES, TRAIN_FRACTION, (ORIGINAL_IMAGE_WIDTH, ORIGINAL_IMAGE_HEIGHT), (PATCH_SIZE, PATCH_SIZE), (PATCH_TRANSLATION, PATCH_TRANSLATION), ANGLE_STEP, FLIP)
    print("Predicted training dataset size: {0}".format(train_dataset_size))

    dataset = create_dataset(ROOT_DIR, TOTAL_IMAGES, TRAIN_FRACTION, rotation_angle=ANGLE_STEP, flip=FLIP)
    patches = extract_patches(dataset, PATCH_SIZE, PATCH_TRANSLATION)

    def dont_extract(input):
        return input

    input_features = compute_input_features(patches[0:2], dont_extract) # train_img_patches and test_img_patches
    output_features = compute_output_features(patches[2:4], value_to_2d_class, FOREGROUND_THRESHOLD) # train_gt_img_patches and test_gt_img_patches

    from skimage import img_as_float

    X_train = img_as_float(input_features[0])
    Y_train = output_features[0].astype(np.float32)
    X_test = img_as_float(input_features[1])
    Y_test = output_features[1].astype(np.float32)

    # **************** DEFINE THE MODEL ARCHITECTURE *******************

    model = Sequential()

    # Convolution layer with rectified linear activation
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],border_mode='same',input_shape=input_shape))
    model.add(Activation('relu'))

    # Second convolution
    model.add(Convolution2D(nb_filters, kernel_size[1], kernel_size[0]))
    model.add(Activation('relu'))

    model.add(SpatialDropout2D(0.25))

    # Third convolution
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[0]))
    model.add(Activation('relu'))

    # Pooling and dropout
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(SpatialDropout2D(0.25))

    # Full-connected layer
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))

    model.add(Dense(1024))
    model.add(Activation('relu'))

    # Dropout to avoid overfitting
    model.add(GaussianDropout(0.25))

    model.add(Dense(1024))
    model.add(Activation('relu'))

    # Dropout to avoid overfitting
    model.add(Dropout(0.5))

    #Fully-connected layer to ouptut the resulting class
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['fmeasure'])

    # checkpoint
    filepath="models/weights-improvement-{epoch:02d}-{fmeasure:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='fmeasure', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    #class_weight = auto??
    # idx = np.random.permutation(np.arange(X_train.shape[0]))
    # train_size = min(3000000, int(X_train.shape[0]))
    # X_train_small = X_train[idx[:train_size]]
    # Y_train_small = Y_train[idx[:train_size]]


    """
    featurewise_center: Boolean. Set input mean to 0 over the dataset, feature-wise.
    samplewise_center: Boolean. Set each sample mean to 0.
    featurewise_std_normalization: Boolean. Divide inputs by std of the dataset, feature-wise.
    samplewise_std_normalization: Boolean. Divide each input by its std.
    zca_whitening: Boolean. Apply ZCA whitening.
    rotation_range: Int. Degree range for random rotations.
    width_shift_range: Float (fraction of total width). Range for random horizontal shifts.
    height_shift_range: Float (fraction of total height). Range for random vertical shifts.
    shear_range: Float. Shear Intensity (Shear angle in counter-clockwise direction as radians)
    zoom_range: Float or [lower, upper]. Range for random zoom. If a float,  [lower, upper] = [1-zoom_range, 1+zoom_range].
    channel_shift_range: Float. Range for random channel shifts.
    fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}. Points outside the boundaries of the input are filled according to the given mode.
    cval: Float or Int. Value used for points outside the boundaries when fill_mode = "constant".
    horizontal_flip: Boolean. Randomly flip inputs horizontally.
    vertical_flip: Boolean. Randomly flip inputs vertically.
    rescale: rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (before applying any other transformation).    """
    train_datagen = ImageDataGenerator(
 #                           samplewise_center=True,
 #                           samplewise_std_normalization=True,
 #                           zca_whitening=False,
                            rotation_range=90,
                            shear_range=0.2,
                            width_shift_range=0.25,
                            height_shift_range=0.25,
                            horizontal_flip=True,
                            vertical_flip=True)

#    test_datagen = ImageDataGenerator(
#                            samplewise_center=True,
#                            samplewise_std_normalization=True,
#                            zca_whitening=False)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    train_datagen.fit(X_train)
#    test_datagen.fit(X_test)


    # fits the model on batches with real-time data augmentation:
    model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size),
                        samples_per_epoch=train_dataset_size*10, nb_epoch=nb_epoch, class_weight='auto', callbacks=callbacks_list, verbose=1, validation_data=(X_test, Y_test))

#     model.fit_generator(X_train_small, Y_train_small, batch_size=batch_size, nb_epoch=nb_epoch, class_weight='auto', callbacks=callbacks_list, verbose=1, validation_data=(X_test, Y_test))


    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    model.save('models/' + model_name)


if __name__ == '__main__':
    train_cnn()
