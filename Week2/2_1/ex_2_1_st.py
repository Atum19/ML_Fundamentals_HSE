import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier


target_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
              "wine/wine.data")

abalone = pd.read_csv(target_url, header=None)
abalone.columns = ['class'] + ['A' + str(i) for i in range(1, 14)]

X = abalone.drop('class', axis=1)  # drop column 'class'
y = abalone['class']
feature_names = X.columns
standardized_X = preprocessing.scale(X)

kfold = KFold(len(standardized_X), n_folds=5, shuffle=True, random_state=42)

model_knc = KNeighborsClassifier(n_neighbors=29)   # as param set quantity of neighborhoods

results = []

scores = cross_validation.cross_val_score(model_knc, standardized_X, y, cv=kfold)
print('scores:', scores)
print('Folds: %i, mean squared error: %.2f std: %.2f' % (len(scores), np.mean(np.abs(scores)),
                                                         np.std(scores)))
