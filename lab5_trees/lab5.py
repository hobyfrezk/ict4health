# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:18:39 2017

@author: hoby
"""

"""
1.unlike K-mean alg or other clustering alg, you dont have to 
predefined number of clusters as a parameter before you start clustering
2.we can change the number of clusters by our prefer

"""

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import preprocessing
import pandas as pd



def get_from_coords(df, x, y):
    return df.iloc[x][y]
    
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

    Dy = labeled_set[labeled_set==2]
    Tp = predicted_set[predicted_set==2][labeled_set==2]
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
# delete samples with unknown data
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
x1 = pd.DataFrame(preprocessing.scale(df.drop(['class'], axis=1)),columns=df.columns[:-1])
x2= x1.loc[:,["al","sg"]]

#================================
pdf = PdfPages("lab5_plot.pdf")
clusters=pd.DataFrame(columns=["all features","2 features"])
for i,x in enumerate([x1,x2]):
    D = pdist(x, 'euclidean')
    Z = linkage(D, method='single')
    
    # retrieve clusters:
    if i==0: 
        k=2
        clusters.loc[:,"all features"] = fcluster(Z,k,criterion='maxclust')   
    else:
        k=2
        clusters.loc[:,"2 features"] = fcluster(Z,k,criterion='maxclust')   

    # show dendrogram
    plt.figure(figsize=(80,30))
    if i==0:
        plt.title('Hierarchical Clustering Dendrogram',fontsize=50)
    else:
        plt.title('Hierarchical Clustering Dendrogram only 2 features',fontsize=50) 
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=16)
    plt.savefig(pdf, format='pdf')
    plt.show()
    plt.close()
    
    #Truncate dendrogram
    plt.figure(figsize=(80,30))
    if i==0:
        plt.title('Truncated Hierarchical Clustering Dendrogram',fontsize=50)
    else:
        plt.title('Truncated Hierarchical Clustering Dendrogram with only 2 features',fontsize=50)
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        truncate_mode='lastp',
        p=12,
        show_contracted=True
    )
    plt.savefig(pdf, format='pdf')
    plt.close()
pdf.close()

class_id=df.loc[:,"class"]
class_id.index=range(158)
evaluation=result(class_id,clusters.loc[:,"2 features"])
evaluation.to_csv("evaluation.csv")
#================================
# classification
feat=()
for i in range(50000):
    X = x1.as_matrix()
    Y = list(df["class"].values)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X,Y)
    Y_pred = clf.predict(X)
    feat_importance=pd.Series(clf.feature_importances_,index=df.columns[:-1])
    feat= feat+(np.setdiff1d(np.array(feat_importance.nonzero()),np.array(3))[0],)
#================================
# analysis
f=list(set(feat))
count=pd.DataFrame(index=f,columns=["count", "importance"])
for i,f0 in enumerate(f):
    count.loc[f0,"count"]=feat.count(f0)
count.loc[:,"importance"]=0.0317
count.index=df.columns[f]
count.to_csv('feature appearance count.csv')

#================================
# accuracy, sensitivity, specificity
# - P (Tp |Dy ) is the test sensitivity (true positive rate)
# - P (Tn |Dn ) is the test specificity (true negative rate)
accuracy=np.mean(Y_pred == Y) * 100
