import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer


words = []
newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space'])
vectorizer = TfidfVectorizer(min_df=1)

texts = newsgroups.data
numb_clas = newsgroups.target

y = numb_clas
X = vectorizer.fit_transform(texts)

feature_mapping = vectorizer.get_feature_names()

model = svm.SVC(C=1.0, kernel='linear', random_state=241) 
model.fit(X, y)

feature_names = np.asarray(feature_mapping)
coef_0 = model.coef_.toarray()[0]
values = abs(coef_0)
top10 = np.argsort(values)[-10:]
for i in range(10):
    words.append(feature_mapping[top10[i]])
    
words.sort()
print('-->', words)
