# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    w = np.array([np.dot(np.dot(np.linalg.inv(np.dot(tx.T, tx)), tx.T), y)]).T
    mse = compute_loss(y, tx, w)
    return mse, w
    
# x = np.array([[1,2],[3,4],[5,6]])
# y = np.array([5,4,3])
# least_squares(y,x)