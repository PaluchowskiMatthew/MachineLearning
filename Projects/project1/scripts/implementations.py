# -*- coding: utf-8 -*-
"""some user helper machine learning functions. (mostly copied from previous labs)"""
import numpy as np
from costs import *
from helpers import *
import matplotlib.pyplot as plt
from proj1_helpers import batch_iter
from proj1_helpers import predict_labels
from IPython.core.debugger import Tracer

'''
        BASE FUNCTIONS
'''

'''
        Least Squares and Ridge Regression
'''

def least_squares(y, tx):
    """calculate the least squares solution."""
    tx_transposed = np.transpose(tx)
    w = np.linalg.solve(np.dot(tx_transposed, tx), np.dot(tx_transposed, y))
    rmse = compute_RMSE(y, tx, w)
    return w, rmse


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    N = tx.shape[0]
    M = tx.shape[1]
    w = np.linalg.solve(np.dot(tx.T, tx) + (2*N*lambda_)*np.identity(M), np.dot(tx.T, y))
    rmse = compute_RMSE(y, tx, w)
    return w, rmse
# ***************************************************
#           Lazare's Implementetion
# ***************************************************
#   Both methods are equal in terms of returned w
# def ridge_regression(y, tx, lamb):
#     """implement ridge regression."""
#     tx_transposed = np.transpose(tx)
#     lambda2 = lamb*2*len(y)
#     txxt = np.dot(tx_transposed, tx)
#     diag = lambda2*np.ones(txxt.shape[0])
#     lambIm = np.diagflat(diag)
#     w = np.linalg.solve(txxt+lambIm, np.dot(tx_transposed, y))
#     return w



def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent
    Return loss and weights.
    """
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - (gamma * gradient)
    rmse = compute_RMSE(y, tx, w)
    return w, rmse


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Compute a stochastic gradient for batch data."""
    batch_size = 1
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size/batch_size))
    w = initial_w
    for n_iter in range(max_iters):
        gradient_sum = np.zeros(tx.shape[1])
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad = compute_gradient(minibatch_y, minibatch_tx, w)
            gradient_sum += grad

        gradient = (1/num_batches_max) * gradient_sum
        w = w - (gamma * gradient)
    rmse = compute_RMSE(y, tx, w)
    return w, rmse


'''
        Logistic Regression
'''


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent.
    Return the loss and the updated w.
    """
    w = initial_w # np.zeros((tx.shape[1],1))
    for iter in range(max_iters):
        gradient = calculate_logistic_gradient(y, tx, w)
        w = w - gamma * gradient
    #loss = calculate_loss_by_likelyhood(y, tx, w)
    loss = compute_RMSE(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w , max_iters, gamma):
    """
    Penalized logistic regression using gradient descent.
    Return the loss and the updated w.
    """
    w = initial_w # np.zeros((tx.shape[1],1))
    for iter in range(max_iters):
        gradient, hessian = calculate_penalized_logistic_gradient(y, tx, w, lambda_)
        w = w - gamma * np.linalg.solve(hessian, gradient)

    loss_penalty = lambda_ * np.sum(np.power(w, 2))
    #loss = calculate_loss_by_likelyhood(y, tx, w) + loss_penalty
    loss = compute_RMSE(y, tx, w)
    return w, loss

'''
        Additonal Functions
'''

def learning_by_newton_method(y, tx, initial_w, max_iters, gamma):
    """
    Do logistic regriession by Newton's method.
    return the loss and updated w.
    """
    w = initial_w
    for iter in range(max_iters):
        grad = calculate_logistic_gradient(y, tx, w)
        hess = calculate_hessian(y, tx, w)
        if (np.linalg.matrix_rank(hess)==hess.shape[0]):
            w = w - np.dot(gamma, np.linalg.solve(hess, grad))
    rmse = compute_RMSE(y, tx, w)
    return w, rmse

def learning_by_IRLS(y, tx, initial_w, max_iters):
    '''
    Iteratively reweighted least squares
    '''
    w = initial_w
    for iter in range(max_iters):
        sig = sigmoid(np.dot(tx, w))
        S_diag = np.multiply(sig, 1-sig)
        S = np.diagflat(S_diag)
        z = np.dot(tx, w)+np.linalg.solve(S, y-sig)

        txt_S = np.dot(np.transpose(tx), S)
        txt_S_tx = np.dot(txt_S, tx)
        txt_S_z = np.dot(txt_S, z)
        w = np.linalg.solve(txt_S_tx,txt_S_z)
    rmse = compute_RMSE(y, tx, w)
    return w, rmse


'''
        HELPER FUNCTIONS
'''

'''
        Least Squares helper functions
'''
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = tx.shape[0]
    normalization = 1/N
    e = np.dot(tx, w) - y
    gradient = np.dot(tx.T, e) * normalization
    return gradient
# ***************************************************
#           Lazare's Implementetion
# ***************************************************
#   Lazare's impelmentation return gradient with minus sign
# def compute_gradient_laz(y, tx, w):
#     """Compute the gradient."""
#     e = compute_loss(y, tx, w)
#     deltaL = -np.dot(np.transpose(tx), e)/len(y)
#     return deltaL


'''
        Logistic Regression helper functions
'''

def sigmoid(x):
    """apply NUMERICALLY STABLE sigmoid function on x."""
    # ***************************************************
    # Lazare uses vectorized version
    # ***************************************************
    # if x >= 0:
    #     z = np.exp(-x)
    #     return 1 / (1 + z)
    # else:
    #     z = np.exp(x)
    #     return z / (1 + z)
    return 1 / (1 + np.exp(-x))

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # ***************************************************
    # Lazare uses vectorized version
    # ***************************************************
    # sig_function  = np.vectorize(sigmoid)
    # sig = sig_function(np.dot(tx, w))
    sig = sigmoid(np.dot(tx, w))
    s_nn = np.multiply(sig, (1-sig))
    S = np.diagflat(s_nn)
    return np.dot(tx.T, np.dot(S, tx))

def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    # ***************************************************
    # Lazare uses vectorized version
    # ***************************************************
    # sig_function  = np.vectorize(sigmoid)
    # sig = sig_function(np.dot(tx, w))
    sig = sigmoid(np.dot(tx, w))
    return np.dot(tx.T, (sig - y) )


def calculate_penalized_logistic_gradient(y, tx, w, lambd):
    """return the loss, gradient, and hessian."""
    # ***************************************************
    gradient = calculate_logistic_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)

    gradient_penalty = lambd * 2 * w
    hessian_penalty = np.diagflat(2*lambd)
    return gradient + gradient_penalty, hessian+hessian_penalty


##
#   CROSS VALIDATION - TO BE CLEANED
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

def cross_validation_mat(y, x, k_fold, seed, lambd, max_iters=1000):
    """return the loss of ridge regression."""
    loss_tr, loss_te = 0.0, 0.0
    losses_tr, losses_te = [], []
    k_indices = build_k_indices(y, k_fold, seed)
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

        train_weight, train_rmse = least_squares(y_train, x_train)
        #logistic_regression(np.array([y_train]).T, x_train, lambd, max_iters)

                                    #least_squares_GD(y_train, x_train, lambd, max_iters)
                                    #ridge_regression(y_train, x_train, lambd)

        # ***************************************************
        # calculate the loss for train and test data
        # ***************************************************
        #train_weight = train_weight[:,0]
        test_rmse = compute_RMSE(y_test, x_test, train_weight)
        losses_tr = np.append(train_rmse, losses_tr)
        losses_te = np.append(test_rmse, losses_te)
        loss_tr += train_rmse
        loss_te += test_rmse

    plt.plot(range(0,k_fold), losses_tr, marker=".", color='r', label='test error')
    plt.plot(range(0,k_fold), losses_te, marker=".", color='b', label='train error')
    plt.legend()
    print(losses_tr)
    print(losses_te)
    # print(train_weight)
    return loss_tr/k_fold, loss_te/k_fold

def cross_validation_laz(y, x, k_fold, seed, lambd, max_iter):
    """return the loss of ridge regression."""
    gamma = 0.001
    loss_tr, loss_te = 0.0, 0.0
    k_indices = build_k_indices(y, k_fold, seed)
    lambds = np.logspace(-3, 0, 10)
    weight_folds = np.zeros((x.shape[1]+1, 1))
    weights = np.zeros((len(lambds), x.shape[1]+1))
    rmse_tr = []
    rmse_te = []
    i = 0
    for lamb in lambds:
        for fold in range(0, k_fold):
            #Create test and train sets
            y_test, y_train, x_test, x_train = np.array([]), np.array([]), np.empty((0,x.shape[1]), float), np.empty((0,x.shape[1]), float)
            x_train = x[k_indices[fold, :]]
            y_train = y[k_indices[fold, :]]
            x_test = np.delete(x, k_indices[fold, :], axis=0)
            y_test = np.delete(y, k_indices[fold, :], axis=0)
            tx_train = np.c_[np.ones((y_train.shape[0], 1)), x_train]
            tx_test = np.c_[np.ones((y_test.shape[0], 1)), x_test]

            w, loss = reg_logistic_regression(y_train, tx_train, lamb, np.zeros((tx_train.shape[1],1)), max_iter, gamma)
            w = np.reshape(w, tx_train.shape[1])
            weights[fold] = w
            test_mse = compute_loss(y_test, tx_test, w)
            train_mse = compute_loss(y_train, tx_train, w)
            loss_tr += np.sqrt(2*train_mse)
            loss_te += np.sqrt(2*test_mse)
        weight_folds = np.reshape(weight_folds/k_fold, tx_train.shape[1])
        weights[i] = weight_folds
        rmse_tr.append(loss_tr/k_fold)
        rmse_te.append(loss_te/k_fold)
        i += 1
        print("lambda done")
                #test_mse = compute_loss(y_test, tx_test, w)
                #train_mse = compute_loss(y_train, tx_train, w)
                #loss_tr[fold, i] = np.sqrt(2*train_mse)
                #loss_te[fold, i]= np.sqrt(2*test_mse)
        #for iter in range(max_iter):
            #train_rmse, w = learning_by_newton_method(y_train, tx_train, w, lambd)
                                    #ridge_regression(y_train, x_train, lambd)
                                    #least_squares_GD(y_train, x_train, lambd, max_iters)
                                    #ridge_regression(y_train, x_train, lambd)
        # ***************************************************
        # calculate the loss for train and test data
        # ***************************************************
        #test_mse = compute_loss(y_test, tx_test, w)
        #train_mse = compute_loss(y_train, tx_train, w)
        #loss_tr += np.sqrt(2*train_mse)
        #loss_te += np.sqrt(2*test_mse)
        #w = np.reshape(w, tx_train.shape[1])
        #weights[fold] = w

    return weights, loss_tr, loss_te

def cross_validation_LS(y, x, k_fold, seed):
    """return the loss of ridge regression."""
    k_indices = build_k_indices(y, k_fold, seed)
    weights = np.zeros((k_fold, x.shape[1]+1))
    rmse_tr = []
    rmse_te = []

    for fold in range(0, k_fold):
        #Create test and train sets
        y_test, y_train, x_test, x_train = np.array([]), np.array([]), np.empty((0,x.shape[1]), float), np.empty((0,x.shape[1]), float)
        x_test = x[k_indices[fold, :]]
        y_test = y[k_indices[fold, :]]
        x_train = np.delete(x, k_indices[fold, :], axis=0)
        y_train = np.delete(y, k_indices[fold, :], axis=0)
        tx_train = np.c_[np.ones((y_train.shape[0], 1)), x_train]
        tx_test = np.c_[np.ones((y_test.shape[0], 1)), x_test]
        #tx_train, x_tr_mean, x_tr_std = standardize(x_train)
        #tx_test, x_te_mean, x_te_std = standardize(x_test)

        w, loss = least_squares(y_train, tx_train)
        test_mse = compute_loss(y_test, tx_test, w)
        train_mse = compute_loss(y_train, tx_train, w)
        rmse_tr.append(np.sqrt(2*train_mse))
        rmse_te.append(np.sqrt(2*test_mse))

        w = np.reshape(w, tx_train.shape[1])
        weights[fold] = w

    return weights, rmse_tr, rmse_te

def cross_validation_ridge(y, x, lamb, k_fold, seed):
    k_indices = build_k_indices(y, k_fold, seed)
    weights = np.zeros((k_fold, x.shape[1]+1))
    f1 = []
    for fold in range(0, k_fold):
        y_test, y_train, x_test, x_train = np.array([]), np.array([]), np.empty((0,x.shape[1]), float), np.empty((0,x.shape[1]), float)
        x_test = x[k_indices[fold, :]]
        y_test = y[k_indices[fold, :]]
        x_train = np.delete(x, k_indices[fold, :], axis=0)
        y_train = np.delete(y, k_indices[fold, :], axis=0)       

        #tx_train, m, s = standardize(x_train)
        #tx_test, m, s = standardize(x_test)
        tx_train = np.c_[np.ones((y_train.shape[0], 1)), x_train]
        tx_test = np.c_[np.ones((y_test.shape[0], 1)), x_test]

        w_rid, rmse_rid = ridge_regression(y_train, tx_train, lamb)
        y_pred_rid = predict_labels(w_rid, tx_test)
        f1_rid = sum(abs(y_test-y_pred_rid))/(2*len(y_pred_rid))
        f1.append(f1_rid)
        w_rid = np.reshape(w_rid, tx_train.shape[1])
        weights[fold] = w_rid

    return f1, weights