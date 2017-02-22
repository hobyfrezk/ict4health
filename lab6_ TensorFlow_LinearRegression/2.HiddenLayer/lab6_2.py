# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:50:08 2017

@author: hoby
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 10:08:14 2017

@author: hoby
"""

import tensorflow as tf
import pandas as pd
import numpy as np
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
w1=tf.Variable(tf.random_normal(shape=[X.shape[1],18], name="weight"))
b1=tf.Variable(tf.random_normal(shape=[1,18], name="bias"))
a1=tf.matmul(x,w1)+b1
z1=tf.nn.tanh(a1)
w2=tf.Variable(tf.random_normal(shape=[18,10], name="weight"))
b2=tf.Variable(tf.random_normal(shape=[1,10], name="bias"))
z2=tf.matmul(z1,w2)+b2
w3=tf.Variable(tf.random_normal(shape=[10,1], name="weight"))
b3=tf.Variable(tf.random_normal(shape=[1,1], name="bias"))
y=tf.matmul(z2,w3)+b3
lose=np.subtract(y, t)
# optimizer structure
cost=tf.reduce_sum(tf.squared_difference(y,t)) # objective function
optim=tf.train.GradientDescentOptimizer(learning_rate=1e-5 ,name="GradientDecent")
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
aaa={}
for step in range(100000):
    sess.run(train, feed_dict={x: X, t: Y})
    aaa[step] = lose.eval(session=sess, feed_dict={x: X, t: Y})
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


plt.plot(cost_value.loc[:97,"step"],cost_value.loc[:97,"cost"])
plt.xlabel('iteration numbers')
plt.grid(which='major', axis='both')
plt.legend()
plt.savefig("6.1.2.pdf", format='pdf')
plt.show()
