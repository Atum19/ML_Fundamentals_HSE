import numpy as np
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.datasets import load_boston
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsRegressor


temp = 0
val_p = 0
max_scr = 0
results = []

boston = load_boston()

coef_p = np.linspace(1.0, 10.0, 200)
X, y = boston.data, boston.target
standardized_X = preprocessing.scale(X) 
kfold = KFold(len(standardized_X), n_folds=5, shuffle=True, 
                     random_state=42)

for elem in coef_p:
    model_knc = KNeighborsRegressor(n_neighbors=5, weights='distance', p=elem)
    scores = cross_validation.cross_val_score(model_knc, standardized_X, y, 
                                              cv=kfold)
    print('scores: ', scores)
    print('max scr: ', np.max(scores))
    print('abs scr: ', np.abs(scores))
    temp = np.max(np.abs(scores))
    if (max_scr <= temp):
        max_scr = temp
        val_p = elem  
results.append({'p': val_p, 'max_scr': max_scr})
print('result: ', results)
