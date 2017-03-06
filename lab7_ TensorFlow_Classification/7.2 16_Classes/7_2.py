# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 18:47:42 2017

@author: hoby
"""

import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('arrhythmia.dat', sep=",", header=None)
# data preparation
data = data.loc[:, (data != 0).any(axis=0)] # delete all 0 column
data.columns = range(258)
X = pd.DataFrame(preprocessing.scale(data.iloc[:, :-1])) # total data set and normalizing data
class_ID = data[257]
Y = pd.DataFrame(np.zeros([X.shape[0],16]), columns=list(a for a in range(1,17)))
for i in range(len(class_ID)):
    temp = class_ID[i]
    Y.loc[i,temp] = 1
# initial settings:
x=tf.placeholder(tf.float32, shape=X.shape, name="input_place_holder")
t=tf.placeholder(tf.float32, shape=[X.shape[0],16],name="out_place_holder")

# neural netw structure
# w = shape(features number,nodes)
# b = shape(patients number,nodes)
# layer 1:
w1=tf.Variable(tf.random_normal(shape=[X.shape[1],64], name="weight1"))
b1=tf.Variable(tf.random_normal(shape=[1,64], name="bias1"))
a1=tf.matmul(x,w1)+b1
z1=tf.nn.sigmoid(a1)

# layer 2:
w2=tf.Variable(tf.random_normal(shape=[64,16], name="weight2"))
b2=tf.Variable(tf.random_normal(shape=[1,16], name="bias2"))
a2=tf.matmul(a1,w2)+b2
z2=tf.nn.sigmoid(a2)

# prediction layer:
w3=tf.Variable(tf.random_normal(shape=[16,16], name="weight3"))
b3=tf.Variable(tf.random_normal(shape=[1,16], name="bias3"))
a3=tf.matmul(z2,w3)+b3
y=tf.nn.softmax(a3)

loss=tf.reduce_sum(np.subtract(y, t)**2, axis=1, keep_dims=True)
# optimizer structure
cost=tf.reduce_sum(loss) # objective function
optim=tf.train.GradientDescentOptimizer(learning_rate=1e-3,name="GradientDecent")
train=optim.minimize(cost, var_list=[w1,b1,w2,b2,w3,b3])

sess = tf.Session()
# initialize
init=tf.initialize_all_variables()
# run the learning machine
sess.run(init)

sess.run(train, feed_dict={x: X, t: Y})
step=0
cost_value=pd.DataFrame(columns=["step", "cost"])
cost_value=pd.DataFrame([[step,cost.eval(session=sess, feed_dict={x: X, t: Y})]],columns=["step", "cost"]).append(cost_value, ignore_index=True)        
yval=y.eval(feed_dict={x: X, t: Y},session=sess)

for step in range(1,5000000):
    sess.run(train, feed_dict={x: X, t: Y})
    if step % 1000 == 0:
        temp = cost.eval(session=sess, feed_dict={x: X, t: Y})
        print(step, "cost=",temp)
        yval=y.eval(feed_dict={x: X, t: Y},session=sess)
        cost_value=pd.DataFrame([[step,temp]],columns=["step", "cost"]).append(cost_value, ignore_index=True)        

yval=pd.DataFrame(yval)
yval.columns=range(1,17)
yval.to_csv("yval.csv")

cost_value.to_csv("cost value.csv")
Y.to_csv("Y.csv")

plt.plot(cost_value.loc[:,"step"],cost_value.loc[:,"cost"])
plt.xlabel('iteration numbers')
plt.grid(which='major', axis='both')
ax=plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0,1))
plt.legend()
plt.savefig("7.2.2.pdf", format='pdf')
plt.show()


plt.plot(Y, "ro",label='regressand')
plt.plot(np.round(yval),"bx",label='regression')
plt.xlabel('case number')
plt.grid(which='major', axis='both')
plt.legend()
plt.savefig("7.2.1.pdf", format='pdf')
plt.show()
