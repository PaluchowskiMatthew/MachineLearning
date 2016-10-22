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
    # print(sig.shape)
    # print(y.shape)
    # print(tx.T.shape)
    #Tracer()()
    return np.dot(tx.T, (sig - y) )

def logistic_regression(y, tx, gamma, max_iter):
    """
    Logistic regression using gradient descent.
    Return the loss and the updated w.
    """
    # start the logistic regression
    w = np.zeros((tx.shape[1],1))
    for iter in range(max_iter):
        # get loss and update w.
        # ***************************************************
        # compute the cost
        #loss = calculate_loss_by_likelyhood(y, tx, w)
        # ***************************************************
        # compute the gradient
        gradient = calculate_logistic_gradient(y, tx, w)
        # ***************************************************
        # update w
        # Tracer()()
        w = w - gamma * gradient
    #print(gradient)
    # loss = calculate_loss_by_likelyhood(y, tx, w)
    loss = compute_loss(y, tx, w)
    return loss, w

def penalized_logistic_regression(y, tx, w, lambd):
    """return the loss, gradient, and hessian."""
    # ***************************************************
    # return loss, gradient, and hessian
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    loss_penalty = lambd * np.sum(np.power(w, 2))
    gradient_penalty = lambd * 2 * w
    return loss + loss_penalty, gradient + gradient_penalty, hessian

def reg_logistic_regression(y, tx, lambd , gamma, max_iters):
    """
    Penalized logistic regression using gradient descent.
    Return the loss and the updated w.
    """
    threshold = 1e-8
    losses = []

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, gradient, hessian = penalized_logistic_regression(y, tx, w, lambda_)
        # ***************************************************
        # update w
        w = w - gamma * np.dot(np.linalg.inv(hessian), gradient)
        # log info
        if iter % 500 == 0:
            print("Current iteration={i}, the loss={l}, w={w}".format(i=iter, l=loss, w=w))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        return loss, w
