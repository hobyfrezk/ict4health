# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:18:39 2017

@author: hoby
"""

"""
1.unlike K-means alg or other clustering alg, you dont have to
predefined number of clusters as a parameter before you start clustering
2.we can change the number of clusters by our prefer

"""

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np

def get_from_coords(df, x, y):
    return df.iloc[x][y]

import pandas as pd

df = pd.read_csv('disease.csv', sep=",", header=None, )
attributes = pd.read_csv('attributes.csv', sep=",", header=None, )
attributes.columns = ["short_name","full_name", "units"]
df.columns = attributes["short_name"]

#================================
#covert string into numeric value
keylist = ("normal","abnormal","present","notpresent","yes",
"no","good","poor","ckd","notckd","?", " ")
keymap = (0,1,0,1,0,1,0,1,2,1,"NaN","NaN")
NaNList = []
for x in range(df.shape[0]):
    for y in range(df.shape[1]):
        record = get_from_coords(df,x,y)
        if record in keylist:
            df.iloc[x][y] = keymap[keylist.index(record)]
            if df.iloc[x][y] == "NaN":
                NaNList.append(x)
                break
# delete samples with unkown data
df.drop(df.index[NaNList], inplace=True)
#================================
# generate the linkage matrix
'''
linkage method parameters:
    "single": Nearest point algorithm
    "complete": Farthest Point Algorithm 
    "weighted": Weighted average distance
    "centroid": Centroid distance
    default is "single"
'''
x = df.drop(['class'], axis=1)
D = pdist(x, 'euclidean')
Z = linkage(D, method='single')

#================================
# show plot
plt.figure()
dn = dendrogram(Z)
plt.show()


#================================
# classification
X = x.as_matrix()
Y = list(df["class"].values)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
Y_pred = clf.predict(X)


#================================
# accuracy, sensitivity, specificity
# - P (Tp |Dy ) is the test sensitivity (true positive rate)
# - P (Tn |Dn ) is the test specificity (true negative rate)
accuracy=np.mean(Y_pred == Y) * 100
