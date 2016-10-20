# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE / MAE
    # ***************************************************
    normalization = 1/(2*y.shape[0])
    yy = np.array([y]).T

    MSE = normalization * np.sum(np.power( yy - np.dot(tx, w), 2))
    return MSE

def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    w = np.array([np.dot(np.dot(np.linalg.inv(np.dot(tx.T, tx)), tx.T), y)]).T
    mse = compute_loss(y, tx, w)
    return mse, w
