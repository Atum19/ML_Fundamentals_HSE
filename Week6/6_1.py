import math
import pylab
import sklearn
import numpy as np
import pandas as pd
from skimage import img_as_float
from skimage.io import imread
from sklearn.cluster import KMeans


image = imread('parrots.jpg')
# pylab.imshow(image)

img_flt = img_as_float(image)

img_matr = np.reshape(img_flt, (-1, img_flt.shape[2]))
print(img_matr.shape)

clf = KMeans(init='k-means++', random_state=241)

df = pd.DataFrame(img_matr)
df['cluster'] = clf.fit_predict(df)
print(df.shape)

df_mean = df.copy()
df_median = df.copy()
df_mean_groups = df.groupby('cluster').mean()
df_median_groups = df.groupby('cluster').median()
for i in range(3):
    df_mean.ix[df_mean['cluster']==i, [0, 1, 2]] = \
                                df_mean_groups.ix[i].tolist()
    df_median.ix[df_median['cluster']==i, [0, 1, 2]] = \
                                df_median_groups.ix[i].tolist()

x_tr = df.drop(['cluster'], axis=1)
df_mean = df_mean.drop(['cluster'], axis=1)
df_median = df_median.drop(['cluster'], axis=1)
 
MSE_mean=sklearn.metrics.mean_squared_error(df_mean, x_tr)
MSE_median=sklearn.metrics.mean_squared_error(df_median, x_tr)
  
PSNR_mean = ((20 * math.log10(1)) - (10 * math.log10(MSE_mean)))
PSNR_median = ((20 * math.log10(1)) - (10 * math.log10(MSE_median)))

print('PSNR mean: ', PSNR_mean)
print('PSNR median: ', PSNR_median)
