# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 10:27:29 2016

@author: autorun
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score 

data = pd.read_csv('data-logistic.csv')
#print data
print type(data)
data.columns = ['class'] + ['A' + str(i) for i in range(1, 3)]
#X = data.drop(('class'), axis=1)  # Выбрасываем столбец 'class'.
#Y = data['class']
Y = data.as_matrix(columns=[data.columns[0]])  #.ravel()
#X = data.as_matrix(columns=[data.columns(1,2)])
X = data.as_matrix(columns=data.columns[1:])
#print Y
#y = data[0]
#print len(y)

def euclid_distance(x, y):
    sum_val = 0.0
    for i in xrange(0, len(x)):
        sum_val += math.pow(x[i] - y[i], 2)
    return math.sqrt(sum_val)

#for i in xrange(len(Y)):
#    print X[i,0]

def gradient_descent(x, y, w, k=0.1, steps=10000, C=1.0, tol=1e-5):
    l = len(y)
   # print 'x: ', x
    for step in xrange(steps):
        old_w = w.copy()
        #print '==>', old_w
        #print 'w[0]: ', w[0]
        for i in xrange(len(y)):
            w[0] += k*(1.0/l)*y[i]*x[i,0]*(1.0 - (1.0/(1 + np.exp(-y[i]*(old_w[0]*x[i,0] + old_w[1]*x[i,1])))))
            w[1] += k*(1.0/l)*y[i]*x[i,1]*(1.0 - (1.0/(1 + np.exp(-y[i]*(old_w[0]*x[i,0] + old_w[1]*x[i,1])))))
        w[0] -= k*C*old_w[0]
        w[1] -= k*C*old_w[1]
        dst = euclid_distance(w, old_w)
        print 'step: ', step
        #print 'old_w: ', old_w
        #print 'w: ', w
        #print 'eucl dist: ', dst
        if (dst <= tol):
            break
    return w
    
def sigmoid(x, w):
    res = 1.0/(1.0 + np.exp(-(w[0]*x[0]) - w[1]*x[1]))
    return res
    
def calc_vect_a(x, w):
    a = np.zeros(len(x))
    for j in xrange(len(x)):
        a[j] = sigmoid(x[j], w)
    #print 'a: ', a
    return a
   
z = gradient_descent(X, Y, w=np.array([0.0, 0.0]), C=0.0)
print 'z: ', z
#a = sigmoid(X, w)
a = calc_vect_a(X, z)
#print 'a: ', a
print '1-st roc: ', roc_auc_score(Y, a)
#plt.plot(a, Y)
#plt.show()

z1 = gradient_descent(X, Y, w=np.array([0.0, 0.0]), C=10.0)
print 'z1: ', z1
#print 'X: ', X
a1 = calc_vect_a(X, z1)
#print 'a1: ', a1
print '2-nd roc: ', roc_auc_score(Y, a1)


