import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


datafrm = pd.read_csv('gbm-data.csv', sep=',')

grad_boost = GradientBoostingClassifier
learning_rate = [1, 0.5, 0.3, 0.2, 0.1] 

y_trn = datafrm['Activity']
x_trn = datafrm.drop(('Activity'), axis=1)
yf_trn = y_trn.values
xf_trn = x_trn.values

X_train, X_test, y_train, y_test = train_test_split(xf_trn,
                                                    yf_trn,
                                                    test_size=0.8,
                                                    random_state=241)

loss_test = []
for pos, elem in enumerate(learning_rate):
    clf = grad_boost(learning_rate=elem, n_estimators=250, random_state=241, verbose=True)
    clf.fit(X_train, y_train)
    for i, y_decision in enumerate(clf.staged_decision_function(X_test)):
        y_pred = 1.0 / (1.0 + np.exp(-y_decision))
        loss = log_loss(y_test, y_pred)  
        print('===>>', loss)
        loss_test.append(loss)
    fig = plt.figure()
    ax = fig.add_subplot(111)    
    ax.plot(loss_test, 'r', linewidth=math.sqrt(elem*20.0))            
    min_loss = min(loss_test)
    it = loss_test.index(min_loss)   # iteration number
    
    ax.annotate(str(learning_rate)+'['+str(it)+'] loss='+str(min_loss),
                xy=(it, min_loss),
                xytext=(it+10, min_loss-0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=0.05))
