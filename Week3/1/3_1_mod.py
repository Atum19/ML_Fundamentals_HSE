import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm


svm_data = pd.read_csv('svm_data.csv')
svm_data.columns = ['class'] + ['A' + str(i) for i in range(1, 3)]


X = svm_data.drop('class', axis=1)  # drop column 'class'
y = svm_data['class']
print(X)
print('X:', len(X))
print('y:', len(y))

model = svm.SVC(C=100000.0, kernel='linear', random_state=241) 

model.fit(X, y)
print(model.intercept_[0])

w = model.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0, 1)
yy = a * xx - (model.intercept_[0]) / w[1]

""" plot the parallels to the separating hyperplane that pass
    through the support vectors """
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=80, facecolors='none')

plt.axis('tight')
plt.grid()
plt.show()
