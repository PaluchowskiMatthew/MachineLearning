import numpy as np
import datetime
from costs import *
from functions import *
import matplotlib.pyplot as plt
from proj1_helpers import batch_iter
from IPython.core.debugger import Tracer

def logistic_regression_dataset_gammas_test(y, y_test, train_dataset, test_dataset, max_iters, gammas, dataset_name, figure_id):
    train_losses = []
    test_losses = []
    weights = np.empty((0, train_dataset.shape[1]), float)
    for gamma in np.nditer(gammas):

        start_time = datetime.datetime.now()
        logistic_regression_loss, logistic_regression_w = logistic_regression(np.array([y]).T, train_dataset, gamma, max_iters)

        # Print result
        logistic_regression_loss = compute_rmse_loss(logistic_regression_loss)
        train_losses = np.append(train_losses, logistic_regression_loss)
        test_mse = compute_loss(y_test, test_dataset, logistic_regression_w[:,0])
        test_rmse = np.sqrt(2*test_mse)
        test_losses = np.append(test_losses, test_rmse)

        end_time = datetime.datetime.now()
        exection_time = (end_time - start_time).total_seconds()

        weights = np.vstack((weights, logistic_regression_w.T))
        print("Logistic Regression for {dn}: execution time={t:.3f} seconds. Test RMSE Loss={l}, Train RMSE Loss={tl}".format(dn = dataset_name, t=exection_time, l=test_rmse, tl=logistic_regression_loss))

    plt.figure(figure_id)
    plt.semilogx(gammas, test_losses, marker=".", color='r', label='test error')
    plt.semilogx(gammas, train_losses, marker=".", color='b', label='train error')
    plt.title(dataset_name)
    plt.xlabel("gamma")
    plt.ylabel("rmse")
    plt.grid(True)
    plt.legend()
    return train_losses, test_losses, weights

def logistic_regression_dataset_single_gamma_test(y, y_test, train_dataset, test_dataset, max_iters, gamma, dataset_name):
    start_time = datetime.datetime.now()
    logistic_regression_loss, logistic_regression_w = logistic_regression(np.array([y]).T, train_dataset, gamma, max_iters)

    # Print result
    logistic_regression_loss = compute_rmse_loss(logistic_regression_loss)
    test_mse = compute_loss(y_test, test_dataset, logistic_regression_w[:,0])
    test_rmse = np.sqrt(2*test_mse)

    end_time = datetime.datetime.now()
    exection_time = (end_time - start_time).total_seconds()

    print("Logistic Regression for {dn}: execution time={t:.3f} seconds. Train RMSE Loss={l} Test RMSE LOSS={tsl}".format(dn = dataset_name, t=exection_time, l=logistic_regression_loss, tsl=test_rmse))
    return logistic_regression_loss, test_rmse, logistic_regression_w[:,0]

def ridge_regression_dataset_lamdas_test(y, y_test, train_dataset, test_dataset, lambdas, dataset_name, figure_id):
    train_losses = []
    test_losses = []
    start_time = datetime.datetime.now()
    for lamb in np.nditer(lambdas):

        ridge_regression_loss, ridge_regression_gradient_w = ridge_regression(y, train_dataset, lamb)

        ridge_regression_loss = compute_rmse_loss(ridge_regression_loss)
        train_losses = np.append(train_losses, ridge_regression_loss)

        test_mse = compute_loss(y_test, test_dataset, ridge_regression_gradient_w)
        test_rmse = np.sqrt(2*test_mse)
        test_losses = np.append(test_losses, test_rmse)

    end_time = datetime.datetime.now()
    exection_time = (end_time - start_time).total_seconds()
    print("Ridge Regression for {dn}: execution time={t:.3f} seconds. Test RMSE Loss={l}, Train RMSE Loss={tl}".format(dn = dataset_name, t=exection_time, l=test_rmse, tl=ridge_regression_loss))

    plt.figure(figure_id)
    plt.semilogx(lambdas, train_losses, marker=".", color='b', label='Train')
    plt.semilogx(lambdas, test_losses, marker=".", color='r', label='Test')
    plt.title(dataset_name)
    plt.xlabel("gamma")
    plt.ylabel("rmse")
    plt.grid(True)
    plt.legend()
