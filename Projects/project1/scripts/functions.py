# -*- coding: utf-8 -*-
"""some user helper machine learning functions. (mostly copied from previous labs)"""
import numpy as np
from costs import *

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    phi = np.zeros((len(x),degree+1))
    phi[:, 0] = 1
    for i in range(degree):
        phi[:, i+1] = np.multiply(phi[:, i],x)
    
    return phi

def compute_loss(y, tx, w):
    """Calculate the loss."""

    e = y - np.dot(tx, w)
    return e

def least_squares(y, tx):
    """calculate the least squares solution."""
    tx_transposed = np.transpose(tx)
    w = np.linalg.solve(np.dot(tx_transposed, tx), np.dot(tx_transposed, y))
    e = compute_loss(y, tx, w)
    mse = np.dot(np.transpose(e), e)/(2*len(y))
    return w, mse

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    tx_transposed = np.transpose(tx)
    lambda2 = lamb*2*len(y)
    txxt = np.dot(tx_transposed, tx)
    diag = lambda2*np.ones(txxt.shape[0])
    lambIm = np.diagflat(diag)
    w = np.linalg.solve(txxt+lambIm, np.dot(tx_transposed, y))
    return w

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


def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size/batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def tensor_broadcasting(p, q):
    """distance between points, see ex1"""
    return np.sqrt(np.sum((p[:,np.newaxis,:]-q[np.newaxis,:,:])**2, axis=2))

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = compute_loss(y, tx, w)
    deltaL = -np.dot(np.transpose(tx), e)/len(y)
    return deltaL

def gradient_descent(y, tx, initial_w, max_iters, gamma): 
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        deltaL = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        
        w = w - np.dot(gamma, deltaL)
        
        ws.append(np.copy(w))
        losses.append(loss)

    return losses, ws

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    index = np.random.randint(0, len(y))
    s_grad = np.square(y[index] - np.dot(np.transpose(tx[index, :]), w))
    return s_grad

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    L=0
    for n_iter in range(max_epochs):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            L += np.square(y[minibatch_y]-np.dot(tx[minibatch_tx]*w))
        
        w = w - np.dot(gamma, deltaL)
    
        ws.append(np.copy(w))
        losses.append(loss)
        
    return losses, ws

# ************* LOGISTIC REGRESSION ************************************************
def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    sig = sigmoid(np.dot(tx, w))
    S_diag = np.multiply(sig, 1-sig)
    S = np.diagflat(S_diag)
    H = np.dot(np.transpose(tx), np.dot(S,tx))
    return H

def sigmoid(t):
    """apply sigmoid function on t."""
    e_t = np.exp(t)
    sig = e_t/(1+e_t)
    return sig

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    sig = sigmoid(np.dot(tx, w))
    log_like = (-np.dot(np.transpose(y), np.log(sig))+np.dot(np.transpose(1-y),np.log(1-sig)))/len(y)
    return log_like

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    L_grad = np.dot(np.transpose(tx), sigmoid(np.dot(tx, w))-y)
    return L_grad

def learning_by_gradient_descent(y, tx, w, alpha):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    L_grad = calculate_gradient(y, tx, w)
    w = w - np.dot(alpha, L_grad)
    return loss, w

def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    hess = calculate_hessian(y, tx, w)
    return loss, grad, hess

def learning_by_newton_method(y, tx, w, alpha):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, grad, hess = logistic_regression(y, tx, w)
    w = w - np.dot(alpha, np.linalg.solve(hess, grad))
    return loss, w

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    shuffled_x = np.random.permutation(x)
    np.random.seed(seed)
    shuffled_y = np.random.permutation(y)
    
    test_x_size = int(shuffled_x.shape[0] * ratio)
    training_x = shuffled_x[0:test_x_size]
    test_x = shuffled_x[-test_x_size:]
    
    test_y_size = int(shuffled_y.shape[0] * ratio)
    training_y = shuffled_y[0:test_y_size]
    test_y = shuffled_y[-test_y_size:]

    return training_x, test_x, training_y, test_y

""" *************  MATEUSZ FUNCTIONS *********************************************
def compute_gradient(y, tx, w):

    N = y.shape[0]
    yy = np.array([y]).T
    normalization = -1/N
    der_w0 = normalization * np.sum((yy - np.dot(tx, w)))
    der_w1 = normalization * np.dot(np.array([tx[:,1]]), yy - np.dot(tx, w) )[0][0]

    gradient = np.array([[der_w0], [der_w1]])
    return gradient
# compute_gradient(np.array([100, 50]), np.array([[1, 10], [1, 10]]), np.array([[1],[0.1]]))

def least_squares_GD(y, tx, initial_w, max_iters, gamma): 
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
    
    gradient_sum = np.array([[0.0],[0.0]])
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        grad = compute_gradient(minibatch_y, minibatch_tx, w)
        gradient_sum += grad
    return (1/batch_size) * grad
# grad = compute_stoch_gradient(y, tx, np.array([[0.0], [0.0]]), 1)
  
def least_squares_SGD(
        y, tx, initial_w, batch_size, max_iters, gamma):
    
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
"""