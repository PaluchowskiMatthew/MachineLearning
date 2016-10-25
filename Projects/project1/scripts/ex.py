def init():
	#%matplotlib inline
	import numpy as np
	import matplotlib.pyplot as plt
	import zipfile
	import os
	import datetime
	from functions import *
	#%load_ext autoreload
	#%autoreload 2

	# Github does not accept files above 100mb and test.csv is 104mb
	# thus we upload zip whith test.csv which needs to be extracted
	with zipfile.ZipFile("../data/test.csv.zip","r") as zip_ref:
	    zip_ref.extractall("../data/")

	from proj1_helpers import *
	DATA_TRAIN_PATH = '../data/train.csv' # TODO: download train data and supply path here 
	y, x, ids = load_csv_data(DATA_TRAIN_PATH)

	#Lets test some basics: Least Squares Gradient Descent
	from plots import gradient_descent_visualization

def try1():
	# Define the parameters of the algorithm.
	#max_iters = 1
	#gamma = 0.4
	#batch_size = 300
	max_iter = 10000
	threshold = 1e-8
	alpha = 0.001
	ratio = 0.1
	losses = []

	# Initialization
	#w_initial = weights

	#tx = np.c_[np.ones((y.shape[0], 1)), x]
	#w = np.zeros((tx.shape[1], 1))

	x_train, x_test, y_train, y_test = split_data(x, y, ratio)
	tx_train = np.c_[np.ones((y_train.shape[0], 1)), x_train]
	w = np.zeros((tx_train.shape[1], 1))

	# Start gradient descent.
	start_time = datetime.datetime.now()
	#gradient_losses, gradient_ws = least_squares_SGD(y, tX, w_initial, batch_size, max_iters, gamma)
	# start the logistic regression
	for iter in range(max_iter):
	    # get loss and update w.
	    loss, w = learning_by_newton_method(y_train, tx_train, w, alpha)
	    # log info
	    if iter % 1000 == 0:
	        print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
	    # converge criteria
	    losses.append(loss)
	    if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
	        break
	# visualization
	visualization(y_train, x_train, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent")
	print("The loss={l}".format(l=calculate_loss(y, tx, w)))
	end_time = datetime.datetime.now()

	# Print result
	exection_time = (end_time - start_time).total_seconds()
	print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))
