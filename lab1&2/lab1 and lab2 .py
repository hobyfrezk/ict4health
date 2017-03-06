# -*- coding: utf-8 -*-
"""
Created time  gio 01 dic 2016 09:27:31 CET
Finished time mar 06 dic 2016 10:23:35 CET 
@author: WANG
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time


def import_data(F0):
    # F0: feature to be estimated
    data = pd.read_csv('parkinsons_updrs.data', sep=",")
    a = pd.DataFrame(np.zeros(shape=(1,22)), columns=data.columns)
    for patientID in range(1,43):
        patient = data[data['subject#']==patientID]
        patient.loc[:,['test_time']] = patient.loc[:,['test_time']].round(decimals=0)
        for test_time in patient['test_time'].drop_duplicates():
            record = patient[patient['test_time']==test_time]
            if record.shape[0]>2:
                c = pd.DataFrame(record.mean().reshape(1,22),columns=data.columns)
                a = a.append(c,ignore_index=True)
    a = a.drop(a.index[0])
    # set training set and test set from original data set
    data_train = a[a['subject#'] < 37]
    data_test = a[a['subject#'] > 36]
    # standardize data with 0 mean and 1 variance
    data_train_scaled = preprocessing.scale(data_train)
    data_test_scaled = preprocessing.scale(data_test)
    
    # the feature we want to predict
    y_train = data_train_scaled[:, F0]
    X_train = np.delete(np.delete(data_train_scaled, F0, axis = 1), 0, axis = 1)
    y_test = data_test_scaled[:, F0]
    X_test = np.delete(np.delete(data_test_scaled, F0, axis = 1), 0, axis = 1)
    
    return [X_train, X_test,y_train,y_test]
    


def MSE(X_train, X_test,y_train,y_test):
    '''
     MSE algorithm: minimum mean square error
     minimize: 
     ERROR = || y - X*a ||^2 ====>  grad(e(a)) = -2*trans(X)*y + 2*trans(X)*X*a = 0 
     ====> a = [trans(X) * X ]^-1 * trans(X) * y
     ====> [trans(X) * X ]^-1 * trans(X) is called pseudo-inverse matrix, can be computed by np.linalg.pinv()
     notice that ERROR here is a scalar.
    '''
    a_MSE = np.dot(np.linalg.pinv(X_train),y_train)
    y_train_hat = np.dot(X_train, a_MSE)
    y_test_hat = np.dot(X_test, a_MSE)  
    return [y_train_hat,y_test_hat,a_MSE]


def gradient_algorithm(X_train, X_test,y_train,y_test):
    '''
     iterative calculation algorithm:
     The updfer solution requires the computation of inverted matrix, might be 
     too complex for some cases(for example image processing)
     
     we can use iterative solution instead.
     1. give a random initial value of weight vector a
     2. evaluate the gradient 
     3. update vector a towards wherever gives smaller gradient value until a 
        small engough value of grad
     4. gamma, which is called step length/learning rate, is given by us
    '''

    a = np.random.rand(X_train.shape[1],)    
    a_new = np.random.rand(X_train.shape[1],)
    gamma = 1e-4
    error_train = error_test = []

    for i in range(20000):
        a = a_new
        grad = -2*np.dot(X_train.T, y_train) + 2*reduce(np.dot, [X_train.T, X_train, a])
        a_new = a- gamma*grad
        y_train_hat = np.dot(X_train, a_new)
        y_test_hat = np.dot(X_test, a_new)

        error_train.append(metrics.mean_squared_error(y_true=y_train, y_pred= y_train_hat))
        error_test.append(metrics.mean_squared_error(y_true=y_test, y_pred= y_test_hat))
    return [y_train_hat,y_test_hat,error_train,error_test,a_new]
    
    

def steepest_decent_d(X_train, X_test,y_train,y_test):
    '''
     steepest decent algorithm:
     almost same as previous one, the only difference is we use dynamic gamma
     for each step, gamma(step length) = norm(grad)^2/(grad.T*H(a)*grad)
     postfix stands for dynamic gamma
     a_new = a_old - gamma*grad
     '''
     
    a = np.random.rand(20,)
    a_new = np.random.rand(20,)
    H = 4*np.dot(X_train.T, X_train)  
    error_train = error_test = []
    
    for i in range(20000):
        a = a_new
        grad = -2*np.dot(X_train.T, y_train) + 2*reduce(np.dot, [X_train.T, X_train, a])
        gamma = np.square(np.linalg.norm(grad))/(reduce(np.dot, [grad.T, H, grad]))
        a_new = a- gamma*grad
        y_train_hat = np.dot(X_train, a_new)
        y_test_hat = np.dot(X_test, a_new)
        error_train.append(metrics.mean_squared_error(y_true=y_train, y_pred= y_train_hat))
        error_test.append(metrics.mean_squared_error(y_true=y_test, y_pred= y_test_hat))
        
    return [y_train_hat,y_test_hat,error_train,error_test,a_new]
    

def PCR(X_train, X_test,y_train,y_test):
    '''
    - PCR
     1. normailze data, shift data set to 0 mean
     2. compute covariance matrix, np.dot(X_train.T, X_train)/845 is covariance matrix because of 0 mean
     3. compute eigenvalue and eigenvector matrix
     4. sort eigenvector decreasingly and take first L eigenvectors and extract the corresponding eigenvector as 
        column vector as new eigenvector matrix
     5. project our dataset into new reference system
     6. do linear regression after standardizing new data sets (unit variance, 0 mean)
    '''
    
    R = np.dot(X_train.T, X_train)/X_train.shape[0] # covariance matrix, and 845 is number of samples
    w, v = np.linalg.eig(R) # w = eigenvalue, v = unit eigenvector
    
    # - regression with full features
    Z_train_full = preprocessing.scale(np.dot(X_train,v)) # normalized X_train data in new reference system
    Z_test_full = preprocessing.scale(np.dot(X_test, v)) # normalized X_test data in new reference system
    a_full = np.dot(np.linalg.pinv(Z_train_full),y_train)
    y_train_hat_full = np.dot(Z_train_full, a_full)    
    y_test_hat_full = np.dot(Z_test_full, a_full)
    error_train_full = metrics.mean_squared_error(y_true=y_train, y_pred= y_train_hat_full)
    error_test_full = metrics.mean_squared_error(y_true=y_test, y_pred= y_test_hat_full)
    
    L = 0
    # - regression with PCA
    while np.sum(w[0:L]) < 0.9*np.sum(w):
        L = L+1
    v_L = v[:, 0:L]
    Z_train_PCA = preprocessing.scale(np.dot(X_train,v_L)) # normalized X_train data in new reference system
    Z_test_PCA = preprocessing.scale(np.dot(X_test, v_L)) # normalized X_test data in new reference system
    
    a_PCA = np.dot(np.linalg.pinv(Z_train_PCA),y_train)
    
    y_train_hat_PCA = np.dot(Z_train_PCA, a_PCA)
    y_test_hat_PCA = np.dot(Z_test_PCA, a_PCA)    
    error_train_PCA = metrics.mean_squared_error(y_true=y_train, y_pred= y_train_hat_PCA)
    error_test_PCA = metrics.mean_squared_error(y_true=y_test, y_pred= y_test_hat_PCA)
    
    return [y_train_hat_full,y_test_hat_full,y_train_hat_PCA,y_test_hat_PCA,error_train_full,
    error_test_full,error_test_PCA,error_train_PCA,a_full,a_PCA]
    

def MSE_plot(y_train,y_test,y_train_hat, y_test_hat,F0,weights):
    pdf = PdfPages("MSE_plot_"+str(F0+1)+".pdf")

    plt.plot(y_train, y_train_hat)
    plt.title("yhat_train versus y_train")
    plt.xlabel('y_train')
    plt.ylabel('y_hat_train')
    plt.grid(True)
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.plot(y_test, y_test_hat)
    plt.title("yhat_test versus y_test")
    plt.xlabel('y_test')
    plt.ylabel('y_hat_test')
    plt.grid(True)
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.hist(y_train_hat-y_train, 50)
    plt.title("yhat_train - y_train")
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.hist(y_test_hat-y_test, 50)
    plt.title("yhat_test - y_test")
    plt.savefig(pdf, format='pdf')
    plt.close()    
    
    plt.plot(weights)
    plt.title("weights")
    plt.savefig(pdf, format='pdf')
    plt.close()    
    pdf.close()

def gradient_algorithm_plot(y_train,y_test,y_train_hat, y_test_hat,error_train,error_test,F0,weights):
    pdf = PdfPages("gradient_algorithm_plot_"+str(F0+1)+".pdf")
    
    plt.plot(y_train, y_train_hat)
    plt.title("yhat_train versus y_train")
    plt.xlabel('y_train')
    plt.ylabel('y_hat_train')
    plt.grid(True)
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.plot(y_test, y_test_hat)
    plt.title("yhat_test versus y_test")
    plt.xlabel('y_test')
    plt.ylabel('y_hat_test')
    plt.grid(True)
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.hist(y_train_hat-y_train, 50)
    plt.title("yhat_train - y_train")
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.hist(y_test_hat-y_test, 50)
    plt.title("yhat_test - y_test")
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.plot(error_train[0:500])
    plt.title("regression of mean square error of train data set")
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.plot(error_test[0:500])
    plt.title("regression of mean square error of test data set")
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.plot(weights)
    plt.title("weights")
    plt.savefig(pdf, format='pdf')
    plt.close()    
    pdf.close()

def steepest_decent_d_plot(y_train,y_test,y_train_hat, y_test_hat,error_train,error_test,F0,weights):
    pdf = PdfPages("steepest_decent_dynamic_step_plot_"+str(F0+1)+".pdf")

    plt.plot(y_train, y_train_hat)
    plt.title("yhat_train versus y_train")
    plt.xlabel('y_train')
    plt.ylabel('y_hat_train')
    plt.grid(True)
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.plot(y_test, y_test_hat)
    plt.title("yhat_test versus y_test")
    plt.xlabel('y_test')
    plt.ylabel('y_hat_test')
    plt.grid(True)
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.hist(y_train_hat-y_train, 50)
    plt.title("yhat_train - y_train")
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.hist(y_test_hat-y_test, 50)
    plt.title("yhat_test - y_test")
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.plot(error_train[0:500])
    plt.title("regression of mean square error of train data set")
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.plot(error_test[0:500])
    plt.title("regression of mean square error of test data set")
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.plot(weights)
    plt.title("weights")
    plt.savefig(pdf, format='pdf')
    plt.close()    
    pdf.close()



def PCR_plot(y_train,y_test,y_train_hat_full,y_test_hat_full,y_train_hat_PCA,y_test_hat_PCA,error_train_full,
    error_test_full,error_test_PCA,error_train_PCA,F0,weights_full,weights_pca):
    pdf = PdfPages("PCR_plot_"+str(F0+1)+".pdf")
        
    plt.plot(y_train, y_train_hat_full)
    plt.title("yhat_train versus y_train_full")
    plt.xlabel('y_train')
    plt.ylabel('y_hat_train_full')
    plt.grid(True)
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.plot(y_test, y_test_hat_full)
    plt.title("yhat_test versus y_test_full")
    plt.xlabel('y_test')
    plt.ylabel('y_hat_test_full')
    plt.grid(True)
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.plot(y_train, y_train_hat_PCA)
    plt.title("yhat_train versus y_train_PCA")
    plt.xlabel('y_train')
    plt.ylabel('y_hat_train_PCA')
    plt.grid(True)
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.plot(y_test, y_test_hat_PCA)
    plt.title("yhat_test versus y_test_PCA")
    plt.xlabel('y_test')
    plt.ylabel('y_hat_test_PCA')
    plt.grid(True)
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    
    plt.hist(y_train_hat_full-y_train, 50)
    plt.title("yhat_train_full - y_train")
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.hist(y_test_hat_full-y_test, 50)
    plt.title("yhat_test_full - y_test")
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.hist(y_train_hat_PCA-y_train, 50)
    plt.title("yhat_train_PCA - y_train")
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.hist(y_test_hat_PCA-y_test, 50)
    plt.title("yhat_test_PCA - y_test")
    plt.savefig(pdf, format='pdf')
    plt.close()
    
    plt.plot(weights_full)
    plt.title("weights_full")
    plt.savefig(pdf, format='pdf')
    plt.close()    
    
    plt.plot(weights_pca)
    plt.title("weights_pca")
    plt.savefig(pdf, format='pdf')
    plt.close()   
    pdf.close()
        
        
if __name__ == "__main__":
    start_time = time.time()
    '''---- Parameter setting and data import ----'''
    for F0 in [6, 4]:
        [X_train, X_test,y_train,y_test] = import_data(F0)
        
#        '''---- 1. MSE ----'''
#        [y_train_hat,y_test_hat,weights] = MSE(X_train, X_test,y_train,y_test)    
#        MSE_plot(y_train,y_test,y_train_hat, y_test_hat,F0,weights)
#    
    
#        '''---- 2. gradient algorithm ----'''
#        [y_train_hat, y_test_hat,error_train,error_test,weights]=gradient_algorithm(X_train, X_test,y_train,y_test)    
#        gradient_algorithm_plot(y_train,y_test,y_train_hat, y_test_hat,error_train,error_test,F0,weights)

       
        '''---- 3. steepest decent with dynamic step length ----'''
        [y_train_hat, y_test_hat,error_train,error_test,weights]=steepest_decent_d(X_train, X_test,y_train,y_test)    
        steepest_decent_d_plot(y_train,y_test,y_train_hat, y_test_hat,error_train,error_test,F0,weights)
#
#            
#        '''---- 4. PCA Regression ----'''
#        [y_train_hat_full,y_test_hat_full,y_train_hat_PCA,y_test_hat_PCA,error_train_full,
#        error_test_full,error_test_PCA,error_train_PCA,weights_full,weights_pca] = PCR(X_train, X_test,y_train,y_test)
#        PCR_plot(y_train,y_test,y_train_hat_full,y_test_hat_full,y_train_hat_PCA,y_test_hat_PCA,error_train_full,
#        error_test_full,error_test_PCA,error_train_PCA,F0,weights_full,weights_pca)
    
#    '''
#    impliment PCA by using scikit API
#    '''
#    from sklearn.decomposition import PCA
#    L = 8
#    pca = PCA(L)
#    pca.fit(X_train)
#    
#    while sum(pca.explained_variance_ratio_) < 0.9:
#        L = L+1
#        pca = PCA(L)
#        pca.fit(X_train)
#    
#    v_L = pca.components_.T
#    w_L = pca.explained_variance_
#    
#    Z_train_PCA = preprocessing.scale(np.dot(X_train,v_L)) # normalized X_train data in new reference system
#    Z_test_PCA = preprocessing.scale(np.dot(X_test, v_L)) # normalized X_test data in new reference system
#    
#    a_PCA = np.dot(np.linalg.pinv(Z_train_PCA),y_train)
#    y_hat_train_PCA = np.dot(Z_train_PCA, a_PCA)
#    error_train_PCA = metrics.mean_squared_error(y_true=y_train, y_pred= y_hat_train_PCA)
#    y_hat_test_PCA = np.dot(Z_test_PCA, a_PCA)
#    error_test_PCA = metrics.mean_squared_error(y_true=y_test, y_pred= y_hat_test_PCA)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
    

