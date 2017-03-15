import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score 

data = pd.read_csv('data-logistic.csv')
data.columns = ['class'] + ['A' + str(i) for i in range(1, 3)]

X = data.as_matrix(columns=data.columns[1:])
Y = data.as_matrix(columns=[data.columns[0]])


def euclid_distance(x, y):
    sum_val = 0.0
    for i in range(0, len(x)):
        sum_val += math.pow(x[i] - y[i], 2)
    return math.sqrt(sum_val)


def gradient_descent(x, y, w, k=0.1, steps=10000, C=1.0, tol=1e-5):
    l = len(y)
    for step in range(steps):
        old_w = w.copy()
        for i in range(len(y)):
            w[0] += k*(1.0/l)*y[i]*x[i, 0] *\
                    (1.0 - (1.0/(1 + np.exp(-y[i]*(old_w[0]*x[i, 0] + old_w[1]*x[i, 1])))))
            w[1] += k*(1.0/l)*y[i]*x[i, 1] *\
                    (1.0 - (1.0/(1 + np.exp(-y[i]*(old_w[0]*x[i, 0] + old_w[1]*x[i, 1])))))
        w[0] -= k*C*old_w[0]
        w[1] -= k*C*old_w[1]
        dst = euclid_distance(w, old_w)
        print('step: ', step)
        if dst <= tol:
            break
    return w


def sigmoid(x, w):
    res = 1.0/(1.0 + np.exp(-(w[0]*x[0]) - w[1]*x[1]))
    return res


def calc_vect_a(x, w):
    a = np.zeros(len(x))
    for j in range(len(x)):
        a[j] = sigmoid(x[j], w)
    return a

z = gradient_descent(X, Y, w=np.array([0.0, 0.0]), C=0.0)
print('z: ', z)

a = calc_vect_a(X, z)
print('1-st roc: ', roc_auc_score(Y, a))

z1 = gradient_descent(X, Y, w=np.array([0.0, 0.0]), C=10.0)
print('z1: ', z1)

a1 = calc_vect_a(X, z1)
print('2-nd roc: ', roc_auc_score(Y, a1))
