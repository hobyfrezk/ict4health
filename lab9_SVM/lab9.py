# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:26:36 2017

@author: hoby
"""

import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
import time

def preprocessing():
    """
    1. read file from arrhythmia.dat
    2. delete all 0 columns 
    3. turn 16 classes into 2 class, arrhythmia and no-arrhythmia
    4. return data set with the size of N_patientes × N_features
       and a column vector of class id for each patients
    """
    data = pd.read_csv('arrhythmia.dat', sep=",", header=None)
    data = data.loc[:, (data != 0).any(axis=0)] # delete all 0 column
    data.columns = range(258)
    data.loc[data[257] != 1, 257] = 2 # convert unhealthy label into 2
    
    class_id = pd.Series(data[257]) # class of each sample
    y = data.iloc[:, :-1] # total data set
    
    return [y,class_id]

def svm_model(X_train,X_test,y_train,y_test,box_constraint=1,kernel_type="linear"):
    if kernel_type == 'linear':
        clf = svm.SVC(C=box_constraint,kernel=kernel_type)
    elif kernel_type == 'gaussian':
        clf = svm.SVC(C=box_constraint,kernel='rbf',gamma=100)
    clf.fit(X_train, y_train) 
    y_train_hat = pd.Series(clf.predict(X_train))
    y_train_hat.index = X_train.index
    y_test_hat = pd.Series(clf.predict(X_test))
    y_test_hat.index = X_test.index
    
    return [y_train_hat,y_test_hat]


def evaluation(labeled_set, predicted_set):
    '''
    Tp means that the test is positive (the marker is present in the blood), 
    Tn means that the test is negative (the marker is absent).
    Dy means that the person has the disease, 
    Dn means that the person does not have the disease.
    
    P (Tp |Dy ) is the test sensitivity (true positive rate)
    P (Tn |Dn ) is the test specificity (true negative rate)
    
    return: 1. accuracy, 2.sensitivity, 3.specificity
    '''

    accuracy=np.mean(labeled_set == predicted_set) * 100


    Dy = labeled_set[labeled_set==2]
    Tp = predicted_set[predicted_set==2][labeled_set==2]
    Dn = labeled_set[labeled_set==1]
    Tn = predicted_set[predicted_set==1][labeled_set==1]
    P_tp_dy = float(Tp.size)/Dy.size * 100
    P_tn_dn = float(Tn.size)/Dn.size * 100
    
    return [accuracy,P_tp_dy,P_tn_dn]


def cross_validate(X,y,box_cons,kernel):
    """
    scikit has build-in cross_val_score function to give a evaluation of a classifier
    however, this function only evaluate in terms of accuracy, but no information about
    sensitivity and specificity, so in this case, i develop a function only give different
    size of train/test sets, and make evaluations.
    """
    columns = ["k", "accuracy_train","sensitivity_train","specificity_train",
               "accuracy_test","sensitivity_test","specificity_test"]
    performance = pd.DataFrame(np.zeros(shape=[6,7]), columns=columns)
    for i, k in enumerate(range(5,11)):
        X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=float(1)/k)
        
        y_train_hat,y_test_hat=svm_model(X_train,X_test,y_train,y_test,\
            box_constraint=box_cons,kernel_type=kernel)
        
        train_performance = evaluation(labeled_set=pd.Series(y_train), predicted_set=pd.Series(y_train_hat))
        test_performance = evaluation(labeled_set=pd.Series(y_test), predicted_set=pd.Series(y_test_hat))
        performance.iloc[i,:]=[k,]+train_performance+test_performance
    
    return np.mean(performance.iloc[:,1:],axis=0)




if __name__ == "__main__":
    start_time = time.time()
    
    kernel = "gaussian" # change kernel type manually ["gaussian","linear"]

    X,y = preprocessing()
    
    '''    
    The c parameter trades off misclassification of training examples 
    against simplicity of the decision surface. A low C makes the decision 
    surface smooth, while a high C aims at classifying all training 
    examples correctly by giving the model freedom to select more samples 
    as support vectors.
    
    If you have a lot of noisy observations you should decrease k. It 
    corresponds to regularize more the estimation.
    '''
    c_range = np.logspace(-2, 3, 6) # different box constraint to be evaluated
    #c_range = [1]
    if kernel == "linear":
        linear_performance = pd.DataFrame(np.zeros(shape=[6,6]))
        for i, c in enumerate(c_range):
            print 'current c: ', c
            performance_c = cross_validate(X,y,box_cons=c,kernel="linear")
            linear_performance.iloc[:,i]=performance_c.values
            
        linear_performance.columns = list('c='+ str(x) for x in c_range)
        linear_performance.index = performance_c.index
        linear_performance.to_csv("linear_kernel_performance.csv")

    if kernel == "gaussian":
        gaussian_performance = pd.DataFrame(np.zeros(shape=[6,6]))
        for i, c in enumerate(c_range):
            print 'current c: ', c
            performance_c = cross_validate(X,y,box_cons=c,kernel="gaussian")
            gaussian_performance.iloc[:,i]=performance_c.values
        gaussian_performance.columns = list('c='+ str(x) for x in c_range)
        gaussian_performance.index = performance_c.index
        gaussian_performance.to_csv("gaussian_kernel_performance.csv")

    print("--- %s seconds ---" % (time.time() - start_time))
