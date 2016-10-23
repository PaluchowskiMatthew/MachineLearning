# -*- coding: utf-8 -*-
"""some user helper machine learning functions. (mostly copied from previous labs)"""
import numpy as np
from costs import *
from proj1_helpers import batch_iter
from IPython.core.debugger import Tracer

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = tx.shape[0]
    normalization = 1/N
    e = np.dot(tx, w) - y
    gradient = np.dot(tx.T, e) * normalization
    return gradient


def least_squares_GD(y, tx, gamma, max_iters):
    """Linear regression using gradient descent
    Return loss and weights.
    """
    w = np.zeros(tx.shape[1])
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - (gamma * gradient)
    loss = compute_loss(y, tx, w)
    return loss, w


def least_squares_SGD(y, tx, gamma, max_iters):
    """Compute a stochastic gradient for batch data."""
    batch_size = 5000
    w = np.zeros(tx.shape[1])
    for n_iter in range(max_iters):
        gradient_sum = np.zeros(tx.shape[1])
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad = compute_gradient(minibatch_y, minibatch_tx, w)
            gradient_sum += grad

        gradient = (1/batch_size) * grad
        w = w - (gamma * gradient)
    loss = compute_loss(y, tx, w)
    return loss, w

def least_squares(y, tx):
    """calculate the least squares solution."""

    w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T, tx)), tx.T), y)
    mse = compute_loss(y, tx, w)
    return mse, w

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    # ***************************************************
    N = tx.shape[0]
    M = tx.shape[1]
    w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T, tx) + (2*N*lamb)*np.identity(M)), tx.T), y)
    mse = compute_loss(y, tx, w)
    return mse, w

##
#   LOGISTIC REGRESSION
##
def sigmoid(x):
    """apply NUMERICALLY STABLE sigmoid function on x."""
    # ***************************************************
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)
    #return 1 / (1 + np.exp(-t))

def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    # ***************************************************
    sig_function  = np.vectorize(sigmoid)
    sig = sig_function(np.dot(tx, w))
    return np.dot(tx.T, (sig - y) )

def logistic_regression(y, tx, gamma, max_iter):
    """
    Logistic regression using gradient descent.
    Return the loss and the updated w.
    """
    # start the logistic regression
    w = np.zeros((tx.shape[1],1))
    for iter in range(max_iter):
        # ***************************************************
        # compute the gradient
        gradient = calculate_logistic_gradient(y, tx, w)
        # ***************************************************
        # update w
        # Tracer()()
        w = w - gamma * gradient
    loss = calculate_loss_by_likelyhood(y, tx, w)
    # loss = compute_loss(y, tx, w)
    return loss, w

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # ***************************************************
    sig_function  = np.vectorize(sigmoid)
    sig = sig_function(np.dot(tx, w))
    s_nn = sig*(1-sig)
    Tracer()()
    S = np.diag(np.ndarray.flatten(s_nn))
    return np.dot(tx.T, np.dot(S, tx))

def penalized_logistic_regression(y, tx, w, lambd):
    """return the loss, gradient, and hessian."""
    # ***************************************************
    # return loss, gradient, and hessian
    #loss = calculate_loss_by_likelyhood(y, tx, w)
    gradient = calculate_logistic_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    #loss_penalty = lambd * np.sum(np.power(w, 2))
    gradient_penalty = lambd * 2 * w
    return gradient + gradient_penalty, hessian

def reg_logistic_regression(y, tx, lambd , gamma, max_iters):
    """
    Penalized logistic regression using gradient descent.
    Return the loss and the updated w.
    """
    # start the logistic regression
    w = np.zeros((tx.shape[1],1))
    for iter in range(max_iters):
        # get loss and update w.
        gradient, hessian = penalized_logistic_regression(y, tx, w, lambd)
        # ***************************************************
        # update w
        w = w - gamma * np.dot(np.linalg.inv(hessian), gradient)
        # log info
    loss_penalty = lambd * np.sum(np.power(w, 2))
    loss = calculate_loss_by_likelyhood(y, tx, w) + loss_penalty
    return loss, w


##
#   CROSS VALIDATION
##

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_fold, seed, lambd):
    """return the loss of ridge regression."""
    loss_tr, loss_te = 0.0, 0.0
    k_indices = build_k_indices(y, k_fold, seed)
    max_iters=1000
    for fold in range(0, k_fold):
        # ***************************************************
        # get k'th subgroup in test, others in train
        # ***************************************************
        y_test, y_train, x_test, x_train = np.array([]), np.array([]), np.empty((0,x.shape[1]), float), np.empty((0,x.shape[1]), float)
        for row in k_indices:
            kth_subgroup = k_indices[k_indices.shape[0]-1-fold]

            if  np.array_equal(row, kth_subgroup):
                x_test = np.vstack((x_test, x[row]))
                y_test = np.append(y_test, y[row])
            else:
                x_train = np.vstack((x_train, x[row]))
                y_train = np.append(y_train, y[row])

        train_mse, train_weight = ridge_regression(y_train, x_train, lambd)
                                    #least_squares(y_train, x_train)
                                    #least_squares_GD(y_train, x_train, lambd, max_iters)
                                    #ridge_regression(y_train, x_train, lambd)

        # ***************************************************
        # calculate the loss for train and test data
        # ***************************************************
        test_mse = compute_loss(y_test, x_test, train_weight)
        loss_tr += np.sqrt(2*train_mse)
        loss_te += np.sqrt(2*test_mse)

    return train_weight, loss_tr/k_fold, loss_te/k_fold
