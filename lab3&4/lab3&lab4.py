# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:33:23 2016
Finished on Sun Dec 10 15:53:27 2016 
@author: WANG
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import time

global N_class

def compare_dis(x):
    if x[0]<x[1]:
        return 1
    if x[0]>x[1]:
        return 2
        
def compare_prob(x):
    if x[0]<x[1]:
        return 2
    if x[0]>x[1]:
        return 1
            
def distance(x,y):
    sub = np.subtract(x, y)
    return np.linalg.norm(sub)
    
def import_Data():
    data = pd.read_csv('arrhythmia.dat', sep=",", header=None)
    data = data.loc[:, (data != 0).any(axis=0)] # delete all 0 column
    data.columns = range(258)


    if N_class ==2:
        data.loc[data[257] != 1, 257] = 2 # convert unhealthy label 2~16 into 2   
        class_id = data[257] # class of each sample
        y = data.iloc[:, :-1] # total data set  without class_id 
        y_scaled = pd.DataFrame(preprocessing.scale(y), columns=y.columns)
        y1 = y_scaled.loc[(class_id == 1),:] # sample belong to region 1
        y2 = y_scaled.loc[(class_id != 1),:] # sample belong to region 2
        return [class_id,y_scaled,y1,y2]
    
    if N_class ==16:
        class_id = data[257] # class of each sample
        y = data.iloc[:, :-1]
        # return dataset, notice that, index = class-1, because index start from 0
        # for example,to access y1 use: y1 = dataset[0]
        dataset =map(lambda x: y.loc[(class_id == x),:], range(1,17))
        return [class_id,y,dataset]

def Min_dist(class_id,y,*arg):
    '''
    Minimum distance criterion
    1. calculate a representative point(central point) for each decision zone
    2. calculate distance to each point
    3. sample belong to the region with minimum distance
    
    distance can be:
    (1) Euclidian distance
    (2) Normalized Euclidian distance
    (3) Mahalanobis distance
    we use euclidian distance here
    '''
    if N_class == 2:
        [y1,y2]=arg
        # use mean value as representative central point
        x1 = pd.DataFrame(y1.mean(axis = 0).reshape(1,257))
        x2 = pd.DataFrame(y2.mean(axis = 0).reshape(1,257))
        # calculate euclidian distance
        sub1 = pd.DataFrame(y.values-x1.values, columns=y.columns)
        sub2 = pd.DataFrame(y.values-x2.values, columns=y.columns) 
        distance1 = np.linalg.norm(sub1, axis=1)
        distance2 = np.linalg.norm(sub2, axis=1)    
        
        output_MinDist = pd.DataFrame({"euclidian_distance1": pd.Series(distance1),\
        "euclidian_distance2": pd.Series(distance2)})
        output_MinDist["result"] = output_MinDist.apply(compare_dis,axis=1)
        output_MinDist["class_id"]=class_id
    
    if N_class ==16: 
        dataset=arg[0]
        mean = map(lambda x: x.mean(axis = 0).reshape(1,257), dataset)
        output_MinDist = pd.DataFrame()
        for i in range(1,17):
            output_MinDist["distance"+ str(i)] = y.apply(lambda x: np.linalg.norm(x.reshape(1,257)-mean[i-1]),axis=1)
        for i in range(1,17):
            output_MinDist["result"] = output_MinDist.apply(lambda x: x.argmin()[8:], axis = 1)
        output_MinDist["class_id"] = class_id
        
    return output_MinDist


def MAP(class_id,y,*arg):
    '''
     Maximun A Posteriori P = (H = Hk |y = y(n)): 
       - for a given estimation, calculate the max probability of regeion it belongs to
     Maxumim likelihood P = (y = y(n) |H = Hk): 
       - we measured y = u, calculate which region is the probability of getting 
         such measurement is maximum
     Procedure:
         1. PCA
         2. get region representative point
         3. get distance between samples to each representative point
         4. turn distance into probability
    '''
    if N_class==2:
        [y1,y2]=arg
        pi1 = float(y1.shape[0])/(y1.shape[0] + y2.shape[0]) # healthy priori probability
        pi2 = float(y2.shape[0])/(y1.shape[0] + y2.shape[0]) # arrhythmia priori probability
        
        # project all features into a subset features, PCA, normalizing features
        R = np.dot(y.T, y)/452 # covariance matrix, and 845 is number of samples
        w, v = map(lambda x: pd.DataFrame(x),np.linalg.eig(R)) # w = eigenvalue, v = unit eigenvector
        
        F1 = 200 # number of features to be considered, to be optimized.
        vF1 = pd.DataFrame(v.loc[:,0:F1-1])
        
        z = pd.DataFrame(preprocessing.scale(np.dot(y, vF1))) #normalized data in new reference system
        z1 = z.loc[(class_id == 1),:] #samples belong to that region
        z2 = z.loc[(class_id == 2),:]
        
        # mean -> region representative point
        w1 = pd.DataFrame(z1.mean(axis = 0).reshape(1,F1))
        w2 = pd.DataFrame(z2.mean(axis = 0).reshape(1,F1))
        
        # calculate region probability
        norm1 = np.linalg.norm(pd.DataFrame(z.values-w1.values, columns=z.columns), axis = 1)
        norm2 = np.linalg.norm(pd.DataFrame(z.values-w2.values, columns=z.columns), axis = 1) 
        
        p1 = pi1 * np.exp(- np.square(norm1)/2)
        p2 = pi2 * np.exp(- np.square(norm2)/2)
    
        probability1=p1/(p1+p2)
        probability2=p2/(p1+p2)
        
        output_MAP = pd.DataFrame({"MAP_Probability1": pd.Series(probability1), "MAP_Probability2": pd.Series(probability2), "class_id": class_id})
        output_MAP["result"] = output_MAP.apply(compare_prob,axis=1)
        
    if N_class==16:
        dataset=arg[0]
        # priori probability
        priori = pd.Series()
        for i in range(1,17):
            priori[str(i)] = (float(y.loc[(class_id == i),:].shape[0])/(y.shape[0]))
        
        # PCA
        R = np.dot(y.T, y)/452 # covariance matrix, and 845 is number of samples
        w, v = map(lambda x: pd.DataFrame(x),np.linalg.eig(R)) # w = eigenvalue, v = unit eigenvector
        F1 = 200 # number of features to be considered, to be optimized.
        vF1 = pd.DataFrame(v.loc[:,0:F1-1])
        
        z = pd.DataFrame(preprocessing.scale(np.dot(y, vF1))) #normalized data in new reference system
        z_class = list(map(lambda x: z.loc[x.index,:],dataset))

        #mean
        w = list(map(lambda x:pd.DataFrame(x.mean(axis=0).reshape(1,F1)), z_class))
        # distance
        norm = pd.DataFrame(np.zeros(shape=[z.shape[0],16]),columns=range(1,17))
        for i in range(452):
            for k in range(16):
                norm.loc[i,k+1]=np.linalg.norm(z.loc[i,:].reshape(1,F1)-w[k])
        
        # probability
        p = pd.DataFrame(np.zeros(shape=[z.shape[0],16]),columns=range(1,17))
        for k in range(16):
            p.loc[:,k+1]= priori[k] * np.exp(- np.square(norm.loc[:,k+1])/2)
        
        
        output_MAP = pd.DataFrame(np.zeros(shape=[z.shape[0],16]),columns=range(1,17))   
        for i in range(452):
            for k in range(16):
                output_MAP.loc[i,k+1]=p.loc[i,k+1]/np.sum(p.loc[i,:])
            output_MAP.loc[i,"result"] = output_MAP.loc[i,:].argmax()
        
        output_MAP.loc[:,"labeled"] = class_id
        
    return output_MAP
    
    
    
    
def hard_K_means(class_id,y,*arg):
    '''
    hard k-means algorithm
    1.Decision region of X is the one who give the best probability for X belongs to that region
    2. Responisibility is a N*M matrix with value is 0 or 1 for each zone    
    '''
    [y1,y2]=arg
    dataset=pd.DataFrame(columns=["result_label","result_random","class_id"])
    
    # parameters initialization, use the mean/random as initial guess
    x_1=y1.mean(axis = 0).reshape(1,257)
    x_2=y2.mean(axis = 0).reshape(1,257)
    x_labeled=[x_1,x_2]
    
    x_1a=np.random.random_sample((1,257))
    x_2a=np.random.random_sample((1,257))
    x_random=[x_1a,x_2a]
    
    for i,X in enumerate([x_labeled, x_random]):
        [x1,x2]=X
        class_1 , class_2 = pd.DataFrame(columns = range(257)),pd.DataFrame(columns = range(257))
        for step in range(50): 
            # assgnment phase:    
            for sample in y.itertuples():
                index = sample[0]
                features = sample[1:]
                if distance(features, x1)<distance(features, x2):
                    if index not in class_1.index: 
                        class_1.loc[index] = features
                        if index in class_2.index:
                            class_2 = class_2.drop(index)
                else:
                    if index not in class_2.index: 
                        class_2.loc[index] = features
                        if index in class_1.index:
                            class_1 = class_1.drop(index)
            # update phase
            x1 = class_1.mean(axis = 0).reshape(1,257)
            x2 = class_2.mean(axis = 0).reshape(1,257)
                                
        class_1['Responsibility'] = 1
        class_2['Responsibility'] = 2
        data = pd.concat([class_1, class_2])
        data.sort_index(inplace=True)
        if i==0:
            dataset['result_label']=data['Responsibility']
        else:
            dataset['result_random']=data['Responsibility']
    dataset['class_id'] = class_id
        
    return dataset
    
    
    
def soft_K_means(class_id,y,*arg):
    '''
    soft k-means algorithm
    1. Decision region of X is the one who give the best probability for X belongs to that region
    2. Responisibility is a N*M matrix with value is a float number, all distance to each region will make 
       contributes when calculating the centroid point.
    '''
    [y1,y2]=arg
    dataset=pd.DataFrame(columns=["result_label","result_random","class_id"])
    beta = 1 # stiffness
    
    # parameters initialization, use the mean/random as initial guess
    x_1=y1.mean(axis = 0).reshape(1,257)
    x_2=y2.mean(axis = 0).reshape(1,257)
    x_labeled=[x_1,x_2]
    
    x_1a=np.random.random_sample((1,257))
    x_2a=np.random.random_sample((1,257))
    x_random=[x_1a,x_2a]
    
    for flag,X in enumerate([x_labeled, x_random]):
        [x1,x2]=X
        r_k=pd.DataFrame(index=range(452), columns=range(2))
        for step in range(20): 
            # assgnment phase:    
            for i in range(452):
                d1=distance(x1,y.iloc[i,:].reshape(1,-1))
                d2=distance(x2,y.iloc[i,:].reshape(1,-1))
                r_k.loc[i,0]=np.exp(-beta*d1)/(np.exp(-beta*d1)+np.exp(-beta*d2))
                r_k.loc[i,1]=np.exp(-beta*d2)/(np.exp(-beta*d1)+np.exp(-beta*d2))

            # update phase
            sum1=pd.DataFrame(np.zeros([1,257]))
            sum2=pd.DataFrame(np.zeros([1,257]))
            r_sum=np.sum(r_k)
            for i in range(452):            
                sum1=sum1+y.loc[i,:].reshape(1,-1)*r_k[0][i]
                sum2=sum2+y.loc[i,:].reshape(1,-1)*r_k[1][i]
                
            x1 = sum1/r_sum[0]
            x2 = sum2/r_sum[1]
    
        for i in range(452):
            r_k.loc[i,"result"] = (int(r_k.loc[i,:].argmax())+1)
        
        if flag==0:
            dataset['result_label']=r_k['result']
        else:
            dataset['result_random'] =r_k['result']
        
        
    dataset.loc[:,"class_id"]=class_id
    return dataset    
    
    
    
    
    
def result(labeled_set,predicted_set):
    '''
    Tp means that the test is positive (the marker is present in the blood), 
    Tn means that the test is negative (the marker is absent).
    Dy means that the person has the disease, 
    Dn means that the person does not have the disease.
    
    P (Tp |Dy ) is the test sensitivity (true positive rate)
    P (Tn |Dn ) is the test specificity (true negative rate)
    False negatives occur with probability P (Tn |Dy ) = 1 − P (Tp |Dy )(one minus the sensitivity); 
    false positives occur with probabilities P (Tp |Dn ) = 1 − P (Tn |Dn ) (one minus the specificity).
    return: 1. accuracy, 2.sensitivity, 3.specioutput_MinDist.loc[:,"class_id"]ficity    
    '''
    predicted_set = pd.to_numeric(predicted_set)
    accuracy=np.mean(labeled_set == predicted_set) * 100
    if N_class==2:
        Dy = labeled_set[labeled_set==2]
        Tp = predicted_set[predicted_set==2][labeled_set==2]
        Dn = labeled_set[labeled_set==1]
        Tn = predicted_set[predicted_set==1][labeled_set==1]
    if N_class==16:
        Dy = labeled_set[labeled_set!=1]
        Tp = predicted_set[predicted_set!=1][labeled_set!=1]
        Dn = labeled_set[labeled_set==1]
        Tn = predicted_set[predicted_set==1][labeled_set==1]        
    P_tp_dy = float(Tp.size)/Dy.size * 100
    P_tn_dn = float(Tn.size)/Dn.size * 100
    
    return pd.Series({
    "accuracy": accuracy,
    "true_positive": P_tp_dy,
    "false_positive":100-P_tp_dy,
    "true_negative": P_tn_dn,
    "false_negative":100-P_tn_dn,
    }) 
    
if __name__ == "__main__":
    start_time = time.time()
    
    N_class=2  #N_class = 2 or 16    
    Lab = 4    #Lab = 3 or 4 (for lab 4, N_class = 2)
    
    
    evaluation=pd.DataFrame(index=["accuracy","true_positive","true_negative","false_positive","false_negative"])

    
    """
    lab3
    """
    if Lab == 3:
        if N_class ==2:
            [class_id,y,y1,y2]=import_Data()
            output_MinDist=Min_dist(class_id,y,y1,y2)
            evaluation["Min Dist"]=result(output_MinDist.loc[:,"class_id"],output_MinDist.loc[:,"result"])

            output_MAP=MAP(class_id,y,y1,y2)
            evaluation["MAP"]=result(output_MAP.loc[:,"class_id"],output_MAP.loc[:,"result"])
    
        if N_class ==16:
            [class_id,y,dataset] = import_Data()
            output_MinDist=Min_dist(class_id,y,dataset)
            evaluation["Min Dist"]=result(output_MinDist.loc[:,"class_id"],output_MinDist.loc[:,"result"])

            output_MAP=MAP(class_id,y,dataset)
            evaluation["MAP"]=result(output_MAP.loc[:,"labeled"],output_MAP.loc[:,"result"])
        
            
        
    '''
    lab4
    '''
    if Lab == 4:
        if N_class ==2:
            [class_id,y,y1,y2]=import_Data()
            output_Hard_K_Means = hard_K_means(class_id,y,y1,y2)
            evaluation["hard K means(labeled)"] = result(output_Hard_K_Means.loc[:,"class_id"],output_Hard_K_Means.loc[:,"result_label"])
            evaluation["hard K means(random)"] = result(output_Hard_K_Means.loc[:,"class_id"],output_Hard_K_Means.loc[:,"result_random"])
            
            output_Soft_K_Means = soft_K_means(class_id,y,y1,y2)  
            evaluation["soft K means(labeled)"] = result(output_Soft_K_Means.loc[:,"class_id"],output_Soft_K_Means.loc[:,"result_label"])
            evaluation["soft K means(random)"] = result(output_Soft_K_Means.loc[:,"class_id"],output_Soft_K_Means.loc[:,"result_random"])
                        
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
    
    
    
    
    
    
    
    
    
