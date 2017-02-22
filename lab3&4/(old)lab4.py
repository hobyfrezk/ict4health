# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 19:11:34 2016

@author: WANG
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing

def distance(x,y):
    sub = np.subtract(x, y)
    return np.linalg.norm(sub)









data = pd.read_csv('arrhythmia_lab3_1.dat', sep="," )
#=============================================================================
# data preparation

data.loc[data['257'] != 1, '257'] = 2 # convert unhealthy label into 2

class_id = data['257'] # class of each sample achieved by MAP algorithm
y = pd.DataFrame(preprocessing.scale(data.iloc[:, :-3])) # Normalize data

y1 = y.loc[(class_id == 1),:] # sample belong to region 1, only includes features
y2 = y.loc[(class_id != 1),:] # sample belong to region 2, only includes features

#=============================================================================


#=============================================================================
'''
hard k-means algorithm
1.Decision region of X is the one who give the best probability for X belongs to that region
2.Probability = (P_priori(y)/(2*pi*sigma2(y))^(N/2)) * exp(-dist(x,y)^2/(2*sigma2(y)))
    - 1. P_priori(y): priori probability for samples blong to region y, P_priori(y) = N_y/N_total
    - 2. sigma2, y are parameters to be iterated
    - 3. y is a row vector same size as x, is a representative point of region y 
    - 4. y = 1/N_y sum (x_y)
    - 5. sigma2 = sum (dist(x,y)^2)

'''
# parameters initialization, use the mean as initial guess
x1 = y1.mean(axis = 0).reshape(1,257)
x2 = y2.mean(axis = 0).reshape(1,257)
class_1 , class_2 = pd.DataFrame(columns = range(257)),pd.DataFrame(columns = range(257))
shape_old = [0,0]


# algorithm start
for i in range(1000): 
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
    
    if shape_old == class_1.shape:
        print 'iteration finished with %d loops' %i
        break
    else:
        shape_old = class_1.shape

if shape_old != class_1.shape:
    print 'infinit loop'

class_1['hard_K_means'] = 1
class_2['hard_K_means'] = 2
data = pd.concat([class_1, class_2])
data['real_class'] = class_id
class_1 = class_2 = None 
#=============================================================================
'''
output:
1. accuracy compared to original doctor's classification
2. distance between MAP centroid and hard K-means centroid with respect to 2 class
'''
accuracy =  np.mean(data['real_class'] == data['hard_K_means']) * 100

x0_1 = y1.mean(axis = 0).reshape(1,257)
x0_2 = y2.mean(axis = 0).reshape(1,257)

distance1 = distance(x0_1, x1)
distance2 = distance(x0_2, x2)
 
print "step 1: use mean as initial guess:"
print "accuracy =", accuracy
print "distance1=" , distance1
print "distance2=" , distance2
print


#=============================================================================
# parameters initialization, use the mean as initial guess
x1 = np.random.random_sample((1,257))
x2 = np.random.random_sample((1,257))
class_1 , class_2 = pd.DataFrame(columns = range(257)),pd.DataFrame(columns = range(257))
shape_old = [0,0]

# algorithm start
for i in range(1000): 
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
    
    if shape_old == class_1.shape:
        print 'iteration finished with %d loops' %i
        break
    else:
        shape_old = class_1.shape

if shape_old != class_1.shape:
    print 'infinit loop'

class_1['hard_K_means'] = 1
class_2['hard_K_means'] = 2
data = pd.concat([class_1, class_2])
data['real_class'] = class_id

#=============================================================================
'''
output:
1. accuracy compared to original doctor's classification
2. distance between MAP centroid and hard K-means centroid with respect to 2 class
'''
accuracy =  np.mean(data['real_class'] == data['hard_K_means']) * 100

x0_1 = y1.mean(axis = 0).reshape(1,257)
x0_2 = y2.mean(axis = 0).reshape(1,257)

distance1 = distance(x0_1, x1)
distance2 = distance(x0_2, x2)
 
print "step 2: use random vector as initial guess:"
print "accuracy =", accuracy
print "distance1=" , distance1
print "distance2=" , distance2
print
