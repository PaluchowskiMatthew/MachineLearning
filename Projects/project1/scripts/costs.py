# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)
    # return calculate_mae(e)

def compute_rmse_loss(mse_loss):
    return np.sqrt(2*mse_loss)

def compute_MSE(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    normalization = 1/(2*y.shape[0])
    yy = np.array([y]).T

    MSE = normalization * np.sum(np.power( yy - np.dot(tx, w), 2))
    return MSE

def calculate_loss_by_likelyhood(y, tx, w):
    """compute the cost by negative log likelihood.
    (Logistic regression)"""
    # ***************************************************
    loss = np.sum(np.log(1+np.exp(np.dot(tx,w))) - np.dot(y.T, np.dot(tx, w)))
    return loss
