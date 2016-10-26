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

#def compute_loss(y, tx, w):
    """Calculate the loss."""
 #   e = y - np.dot(tx, w)
 #   return e

def compute_RMSE(y, tx, w):
    e = y - np.dot(tx, w)
    mse = np.dot(np.transpose(e), e)/(2*len(y))
    rmse = np.sqrt(2*mse)
    return rmse

def least_squares(y, tx):
    """calculate the least squares solution."""
    tx_transposed = np.transpose(tx)
    w = np.linalg.solve(np.dot(tx_transposed, tx), np.dot(tx_transposed, y))
    e = compute_loss(y, tx, w)
    mse = np.dot(np.transpose(e), e)/(2*len(y))
    return w, mse

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
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
    sig = np.zeros((t.shape[0], 1))
    for i in range(len(t)):
        if t[i]>100:    
            sig[i] = 0.99999
        elif t[i] < 100:
            sig[i] = 0.000001
        else:
            sig[i] = 1/(1+np.exp(-t[i]))
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
    mse = compute_loss(y, tx, w)
    rmse = np.sqrt(2*mse)
    L_grad = calculate_gradient(y, tx, w)
    w = w - np.dot(alpha, L_grad)
    return rmse, w

def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    #loss = calculate_loss(y, tx, w)
    mse = compute_loss(y, tx, w)
    rmse = np.sqrt(2*mse)
    grad = calculate_gradient(y, tx, w)
    hess = calculate_hessian(y, tx, w)
    return rmse, grad, hess

def learning_by_newton_method(y, tx, w, alpha):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    rmse, grad, hess = logistic_regression(y, tx, w)
    w = w - np.dot(alpha, np.linalg.solve(hess, grad))
    return rmse, w

def IRLS(y, tx, w):
    sig = sigmoid(np.dot(tx, w))
    S_diag = np.multiply(sig, 1-sig)
    S = np.diagflat(S_diag)
    z = np.dot(tx, w)+np.linalg.solve(S, y-sig)
    loss = calculate_loss(y, tx, w)
    return S, z, loss

def learning_by_IRLS(y, tx, w):
    S, z, loss = IRLS(y, tx, w)
    txt_S = np.dot(np.transpose(tx), S)
    w = np.linalg.solve(np.dot(txt_S, tx),np.dot(txt_S, z))
    return loss, w

def penalized_logistic_regression(y, tx, w, lambd):
    """return the loss, gradient, and hessian."""
    # ***************************************************
    # return loss, gradient, and hessian
    #loss = calculate_loss_by_likelyhood(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    #loss_penalty = lambd * np.sum(np.power(w, 2))
    gradient_penalty = lambd * 2 * w / np.linalg.norm(w)
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
        w = w - gamma * np.linalg.solve(hessian, gradient)
        # log info
    loss_penalty = lambd * np.sum(np.power(w, 2))
    loss = compute_RMSE(y, tx, w) + loss_penalty
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
    
    train_x_size = int(shuffled_x.shape[0] * ratio)
    training_x = shuffled_x[:train_x_size]
    test_x = shuffled_x[(train_x_size):]
    
    train_y_size = int(shuffled_y.shape[0] * ratio)
    training_y = shuffled_y[:train_y_size]
    test_y = shuffled_y[(train_y_size):]

    return training_x, test_x, training_y, test_y

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_fold, seed, lambd, max_iter):
    """return the loss of ridge regression."""
    loss_tr, loss_te = 0.0, 0.0
    k_indices = build_k_indices(y, k_fold, seed)
    max_iters=1000
    weights = np.zeros((k_fold, x.shape[1]+1))
    for fold in range(0, k_fold):
        # ***************************************************
        # get k'th subgroup in test, others in train
        # ***************************************************
        y_test, y_train, x_test, x_train = np.array([]), np.array([]), np.empty((0,x.shape[1]), float), np.empty((0,x.shape[1]), float)
        x_train = x[k_indices[fold, :]]
        y_train = y[k_indices[fold, :]]

        x_test = np.delete(x, k_indices[fold, :], axis=0)
        y_test = np.delete(y, k_indices[fold, :], axis=0)

        tx_train, x_tr_mean, x_tr_std = standardize(x_train)
        tx_test, x_te_mean, x_te_std = standardize(x_test)
        w = np.zeros((tx_train.shape[1], 1))

        for iter in range(max_iter):
            train_rmse, w = learning_by_newton_method(y_train, tx_train, w, lambd)
                                    #ridge_regression(y_train, x_train, lambd)
                                    #least_squares_GD(y_train, x_train, lambd, max_iters)
                                    #ridge_regression(y_train, x_train, lambd)
        # ***************************************************
        # calculate the loss for train and test data
        # ***************************************************
        test_mse = compute_loss(y_test, tx_test, w)
        train_mse = compute_loss(y_train, tx_train, w)
        loss_tr += np.sqrt(2*train_mse)
        loss_te += np.sqrt(2*test_mse)
        w = np.reshape(w, tx_train.shape[1])
        weights[fold] = w

    return weights, loss_tr/k_fold, loss_te/k_fold