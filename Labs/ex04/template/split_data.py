# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)

    shuffled_x = np.random.permutation(x)
    shuffled_y = np.random.permutation(y)
    test_x_size = shuffled_x.shape[0] * ratio
    training_x = shuffled_x[0:test_x_size]
    test_x = shuffled_x[-test_x_size:]
    
    test_y_size = shuffled_y.shape[0] * ratio
    training_y = shuffled_y[0:test_y_size]
    test_y = shuffled_y[-test_y_size:]

    return training_x, test_x, training_y, test_y