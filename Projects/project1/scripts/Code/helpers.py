# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]

    tx = np.hstack((np.ones((x.shape[0],1)), x))
    return tx, mean_x, std_x


def tensor_broadcasting(p, q):
    """distance between points, see ex1"""
    return np.sqrt(np.sum((p[:,np.newaxis,:]-q[np.newaxis,:,:])**2, axis=2))

##
#   Data split for training and test
##

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed

    np.random.seed(seed)
    size = y.shape[0]
    indices = np.random.permutation(size)
    y_test, y_train, x_test, x_train = np.array([]), np.array([]), np.empty((0,x.shape[1]), float), np.empty((0,x.shape[1]), float)
    x_train = x[indices[: int(ratio*size)]]
    y_train = y[indices[: int(ratio*size)]]
    x_test = x[indices[int(ratio*size) :]]
    y_test = y[indices[int(ratio*size) :]]

    return x_train, x_test, y_train, y_test

##
#   Function used for feature combinations
##
import itertools
def combinations(array2d, indeces_list_a, indeces_list_b):
    combinations = list(itertools.product(indeces_list_a, indeces_list_b))
    for comb in combinations:
        new_feature = np.array([array2d[:,comb[0]] * array2d[:,comb[1]]]).T
        array2d = np.hstack((array2d, new_feature))
    return array2d