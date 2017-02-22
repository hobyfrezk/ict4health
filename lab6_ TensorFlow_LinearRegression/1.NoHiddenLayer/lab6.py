# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 10:08:14 2017

@author: hoby
"""

import tensorflow as tf
import pandas as pd

'''
data preparation, saved as processed_data.csv
'''
#from sklearn import preprocessing

#data = pd.read_csv('parkinsons_updrs.data', sep=",")
#tag = data.columns
## modify data into 990 row
#a = pd.DataFrame(np.zeros(shape=(1,22)), columns=data.columns)
#for patientID in range(1,43):
#    patient = data[data['subject#']==patientID]
#    patient.loc[:,['test_time']] = patient.loc[:,['test_time']].round(decimals=0)
#    for time in patient['test_time'].drop_duplicates():
#        record = patient[patient['test_time']==time]
#        if record.shape[0]>2:
#            c = pd.DataFrame(record.mean().reshape(1,22),columns=data.columns)
#            a = a.append(c,ignore_index=True)
#a = a.drop(a.index[0]) 
#a= pd.DataFrame(preprocessing.scale(a), columns=tag ) # normalize data
#
#a.to_csv("processed_data.csv")

'''
training machine, save weights, biases and cost function into csv
'''
a = pd.read_csv('processed_data.csv',sep=",")

#regressors = (u"age","sex","Jitter(%)","Jitter(Abs)","Jitter:RAP","Jitter:PPQ5,"
#"Jitter:DDP","Shimmer","Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","Shimmer:APQ11",
#"Shimmer:DDA","NHR","HNR","RPDE","DFA","PPE")
regressors = (2,3,)+tuple(range(7,23))
regressors = tuple((x - 1) for x in regressors)
X = a.iloc[:,regressors]
Y = pd.DataFrame(a.iloc[:,6])

# initial settings:
tf.set_random_seed(1234)
x=tf.placeholder(tf.float32, shape=X.shape, name="input_place_holder")
t=tf.placeholder(tf.float32, shape=[X.shape[0],1],name="output_place_holder")

# neural netw structure
# w = shape(features number,nodes)
# b = shape(patients number,nodes)
w=tf.Variable(tf.random_normal(shape=[X.shape[1],1], name="weight"))
b=tf.Variable(tf.random_normal(shape=[1,1], name="bias"))
y=tf.matmul(x,w)+b

# optimizer structure
cost=tf.reduce_sum(tf.squared_difference(y,t)) # objective function
optim=tf.train.GradientDescentOptimizer(learning_rate=5e-5,name="GradientDecent")
train=optim.minimize(cost, var_list=[w,b])

sess = tf.Session()
# initialize
init=tf.initialize_all_variables()
# run the learning machine
sess.run(init)

sess.run(train, feed_dict={x: X, t: Y})
step=0
cost_value=pd.DataFrame(columns=["step", "cost"])
cost_value=pd.DataFrame([[step,cost.eval(session=sess, feed_dict={x: X, t: Y})]],columns=["step", "cost"]).append(cost_value, ignore_index=True)        

for step in range(1,20000):
    sess.run(train, feed_dict={x: X, t: Y})
    if step % 1000 == 0:
        temp = cost.eval(session=sess, feed_dict={x: X, t: Y})
        print(step, temp)
        cost_value=pd.DataFrame([[step,temp]],columns=["step", "cost"]).append(cost_value, ignore_index=True)        

#weights = pd.DataFrame(sess.run(w))
#biases = pd.DataFrame(sess.run(b))
#
#weights.to_csv("weights.csv")
#biases.to_csv("biases.csv")
#cost_value.to_csv("cost_value.csv")

'''
regression step
'''
import matplotlib.pyplot as plt
yval=y.eval(feed_dict={x: X, t: Y},session=sess)

plt.plot(Y, "ro",label='regressand')
plt.plot(yval,"bx",label='regression')
plt.xlabel('case number')
plt.grid(which='major', axis='both')
plt.legend()
plt.savefig("6.1.pdf", format='pdf')
plt.show()
