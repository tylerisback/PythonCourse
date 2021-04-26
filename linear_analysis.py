import pandas as pd
import numpy as np
from scipy.stats import t


def the_function(Target, dropped_column, VarOfInterest_name): #y target, x covariates, VarOfInterest is the column of specified interested variable.
    last_data = pd.read_csv('last_data_ready')
    last_data.dropna(axis = 0, inplace = True) #list-wise deletion if it has missing value
    y = np.array(last_data[Target]) #Make them arrays to make calculations
    last_data.drop(columns = [Target], inplace = True)


    ones_ = np.ones(35) #For B0
    X = last_data.copy()
    X['ones'] =ones_
    X.drop(columns = [dropped_column], inplace=True)

    #10 line of  code below is to change the place of ones to column to first place.
    cols = X.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    X = X[cols]
    VarOfInterest = 0
    for i in range(len(X.columns)):
        if X.columns[i] == VarOfInterest_name:
            VarOfInterest = i

    X= np.array(X) #Make them arrays to make calculations

    #Formula : B^= (XX^)^-1 * X^ * y
    B_0, B_1, B_2, B_3 = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)),y)
    all_B = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)),y)

    #y^predictions of linear regression (not future)
    y_predict = []
    for i in range(X.shape[0]):
        y_ = B_0 + B_1 * X[i,1] + B_2 * X[i,2] + B_3 * X[i,3]
        y_predict.append(y_)

    #errors in sample
    errors = []
    for i in range(len(y)):
        e = y_predict[i] - y[i]
        errors.append(e)

    #Total squared errors
    square_errors = []
    for i in errors:
        square_errors.append(i ** 2)

    variance = np.dot(np.transpose(errors), errors)/(len(X[:,0]) - len(X[0,:] -1))

    C = np.linalg.inv(np.dot(np.transpose(X),X)) #C value to calculate standard error
    standard_error_B = []
    for i in range(len(C)):
        standard_error_B.append(np.sqrt(variance * C[i][i]))


    t_confidence_intervals= []
    t_confidence_intervals.append(-t.ppf(0.025, len(X[:,0]) - len(X[0,:] -1)))
    t_confidence_intervals.append(t.ppf(0.025, len(X[:,0]) - len(X[0,:] -1)))


    return all_B, standard_error_B, t_confidence_intervals, VarOfInterest




