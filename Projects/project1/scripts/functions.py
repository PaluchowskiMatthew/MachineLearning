# -*- coding: utf-8 -*-
"""some user helper machine learning functions. (mostly copied from previous labs)"""
import numpy as np
from costs import *

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    yy = np.array([y]).T
    normalization = -1/N
    der_w0 = normalization * np.sum((yy - np.dot(tx, w)))
    der_w1 = normalization * np.dot(np.array([tx[:,1]]), yy - np.dot(tx, w) )[0][0]

    gradient = np.array([[der_w0], [der_w1]])
    return gradient
# compute_gradient(np.array([100, 50]), np.array([[1, 10], [1, 10]]), np.array([[1],[0.1]]))

def least_squares_GD(y, tx, initial_w, max_iters, gamma): 
    """Linear regression using gradient descent"""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - (gamma * gradient)
        ws.append(np.copy(w))
        losses.append(loss)

        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
# least_squares_GD(np.array([100, 50]), np.array([[1, 10],[1, 10]]), np.array([[1], [0.1]]), 1, 1)
    
    
def compute_stoch_gradient(y, tx, w, batch_size):
    """Compute a stochastic gradient for batch data."""
    gradient_sum = np.array([[0.0],[0.0]])
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        grad = compute_gradient(minibatch_y, minibatch_tx, w)
        gradient_sum += grad
    return (1/batch_size) * grad
# grad = compute_stoch_gradient(y, tx, np.array([[0.0], [0.0]]), 1)
  
def least_squares_SGD(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_stoch_gradient(y, tx, w, batch_size)
        loss = compute_loss(y, tx, w)
        w = w - (gamma * gradient)
        ws.append(np.copy(w))
        losses.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws