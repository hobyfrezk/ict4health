# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:57:46 2017

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
data.loc[data[257] != 1, 257] = 2 # convert unhealthy label into 2

Y = pd.DataFrame(data[257]) # class of each sample
Y = Y-1 #convinient with sigmoid activation function
X = pd.DataFrame(preprocessing.scale(data.iloc[:, :-1])) # total data set and normalizing data


# initial settings:
tf.set_random_seed(1234)
x=tf.placeholder(tf.float32, shape=X.shape, name="input_place_holder")
t=tf.placeholder(tf.float32, shape=[X.shape[0],1],name="output_place_holder")

# neural netw structure
# w = shape(features number,nodes)
# b = shape(patients number,nodes)
# layer 1:
w1=tf.Variable(tf.random_normal(shape=[X.shape[1],257], name="weight"))
b1=tf.Variable(tf.random_normal(shape=[1,257], name="bias"))
a1=tf.matmul(x,w1)+b1
# layer 2:
w2=tf.Variable(tf.random_normal(shape=[257,128], name="weight"))
b2=tf.Variable(tf.random_normal(shape=[1,128], name="bias"))
a2=tf.matmul(a1,w2)+b2
z=tf.nn.sigmoid(a2)
# prediction layer:
w3=tf.Variable(tf.random_normal(shape=[128,1], name="weight"))
b3=tf.Variable(tf.random_normal(shape=[1,1], name="bias"))
y=tf.matmul(z,w3)+b3

# optimizer structure
cost=tf.reduce_sum(tf.squared_difference(y,t)) # objective function
optim=tf.train.GradientDescentOptimizer(learning_rate=2.5e-5,name="GradientDecent")
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

for step in range(1,20000):
    sess.run(train, feed_dict={x: X, t: Y})
    if step % 100 == 0:
        temp = cost.eval(session=sess, feed_dict={x: X, t: Y})
        print(step, temp)
        cost_value=pd.DataFrame([[step,temp]],columns=["step", "cost"]).append(cost_value, ignore_index=True)        


yval=y.eval(feed_dict={x: X, t: Y},session=sess)
plt.plot(Y, "ro",label='regressand')
plt.plot(np.round(yval),"bx",label='regression')
plt.xlabel('case number')
plt.grid(which='major', axis='both')
plt.legend()
plt.savefig("7.1.1.pdf", format='pdf')
plt.show()


plt.plot(cost_value.loc[:196,"step"],cost_value.loc[:196,"cost"])
plt.xlabel('iteration numbers')
plt.grid(which='major', axis='both')
plt.legend()
plt.savefig("7.1.2.pdf", format='pdf')
plt.show()
