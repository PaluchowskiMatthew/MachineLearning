import numpy as np
import zipfile
from costs import *
from helpers import *
from proj1_helpers import *
from implementations import *

#***** DOWNLOAD THE DATA **************
with zipfile.ZipFile("../data/test.csv.zip","r") as zip_ref:
    zip_ref.extractall("../data/")

DATA_TRAIN_PATH = '../data/train.csv'
y, x, ids = load_csv_data(DATA_TRAIN_PATH)

#***** FEATURE ENGINEERING *************

#Let's replace the -999 by Nan
x_nan = x.copy()
np.putmask(x_nan, x_nan==-999, np.nan)

#Define the degree of the polynomial basis
degree = 9
x_pow9 = np.zeros((x_nan.shape[0], degree*x_nan.shape[1]))

#Let's build a matrix using a polynomial basis function
for column in range(0,x_nan.shape[1]):
    x_pow9[:, column] = x_nan[:, column]
    for deg in range(1,degree):
        x_pow9[:, column + deg*x_nan.shape[1]] = np.multiply(x_nan[:, column], x_pow9[:, column + (deg-1)*x_nan.shape[1]])

#Adding combinations of features with one another
base_columns = list(range(0, x_nan.shape[1]))
x_pow9 = combinations(x_pow9, base_columns, base_columns)

#Put back specific value instead of Nan
x_pow9[np.isnan(x_pow9)]=-1

#***** CHOOSING THE BEST LAMBDA PARAMETER *******
seed = 2
y = np.reshape(y, (len(y), 1))
#Split the data into 20% training, 80% testing
ratio = 0.3
x_train, x_test, y_train, y_test = split_data(x_pow9, y, ratio, seed)
#Adding a column of 1
tx_train_pow9 = np.c_[np.ones((x_train.shape[0], 1)), x_train]
tx_test_pow9 = np.c_[np.ones((x_test.shape[0], 1)), x_test]

#Setting the interval for lambda
lambds = np.logspace(-7, -2, 100)
best_f1 = 0
best_lamb = 0
k_folds = 10

#Find the best lambda
for lamb in lambds:
    #Learning the weights with Ridge Regression
    w_rid, r = ridge_regression(y_train, tx_train_pow9, lamb)
    #Compute the accuracy
    y_pred_tr = predict_labels(w_rid, tx_train_pow9)
    y_pred_te = predict_labels(w_rid, tx_test_pow9)
    f1_rid_tr = 1-sum(abs(y_train-y_pred_tr))/(2*len(y_pred_tr))
    f1_rid_te = 1-sum(abs(y_test-y_pred_te))/(2*len(y_pred_te))
    #Storing the best lambda
    if f1_rid_te > best_f1:
        best_lamb = lamb
        best_f1 = f1_rid_te

#Do an weight average over cross-fold validation
k_folds = 10
_, w_CV_rid = cross_validation_ridge(y_train, x_train, best_lamb, k_folds, seed)

#Average the weights over folds to obtain more statistically consistent results
w_mean_ridge = w_CV_rid.mean(axis=0)

#***** FEATURE ENGINEERING FOR PREDICTIONS DATSET ********
#Loading the data
DATA_TEST_PATH = '../data/test.csv'
_, X_test, ids_test = load_csv_data(DATA_TEST_PATH)

#Applying same operations as on the train dataset
x_nan_test = X_test.copy()
np.putmask(x_nan_test, x_nan_test==-999, np.nan)
degree = 9
x_TEST = np.zeros((x_nan_test.shape[0], degree*x_nan_test.shape[1]))

#Let's build a matrix using a polynomial basis function (power 9)
for column in range(0,x_nan_test.shape[1]):
    x_TEST[:, column] = x_nan_test[:, column]
    for deg in range(1,degree):
        x_TEST[:, column + deg*x_nan_test.shape[1]] = np.multiply(x_nan_test[:, column], x_TEST[:, column + (deg-1)*x_nan_test.shape[1]])

#Adding combinations of features with one another
base_columns = list(range(0, x_nan_test.shape[1]))
x_TEST = combinations(x_TEST, base_columns, base_columns)

#Put back specific value instead of Nan
x_TEST[np.isnan(x_TEST)]=-1
#Adding a line of 1
tx_TEST = np.c_[np.ones((x_TEST.shape[0], 1)), x_TEST]

#***** COMPUTE PREDICTIONS AND OUTPUT SUBMISSION
#Predictions
y_pred = predict_labels(w_mean_ridge, tx_TEST)
#Ouptut
OUTPUT_PATH = '../data/Group86.csv'
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
