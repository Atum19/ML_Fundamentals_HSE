import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


pca = PCA(n_components=10)

djia_indx = pd.read_csv('djia_index.csv', sep=',')
data_prc = pd.read_csv('close_prices.csv', sep=',')
X = data_prc.ix[:, 1:].as_matrix()

pca.fit(X)
variance_ratio = pca.explained_variance_ratio_
res_sum = 0.0
for pos, elem in enumerate(variance_ratio):
    res_sum += elem
    if res_sum >= 0.9:
        print(res_sum, '==>', pos+1)  # numeration star from 0
        break
    
t_pca = pca.transform(X)
X = [x[0] for x in t_pca]   # value of first component
Y = [x[1] for x in djia_indx.values]  # DJI column
print(np.corrcoef(X, Y))
compn = pca.components_[0]
idx = np.argmax(compn) + 1
print('idx:', idx)
