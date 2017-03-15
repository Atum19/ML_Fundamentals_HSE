# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:51:48 2016

@author: autorun
"""

import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer


words= []
newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )
    
vectorizer = TfidfVectorizer(min_df=1)
#print newsgroups.target_names
#print len(newsgroups.data)
#print len(newsgroups.filenames)
#print newsgroups.target[:1500]
#print vectorizer
#for d in newsgroups:
#    print d

#print newsgroups['filenames']

texts = newsgroups.data
numb_clas = newsgroups.target

X = vectorizer.fit_transform(texts)
y = numb_clas
#print X

feature_mapping = vectorizer.get_feature_names()
#print len(feature_mapping)

model = svm.SVC(C=1.0, kernel='linear', random_state=241) 
model.fit(X, y)
feature_names = np.asarray(feature_mapping)
coef_0 = model.coef_.toarray()[0]
#print len(coef_0)
values = abs(coef_0)
top10 = np.argsort(values)[-10:] 
#print 'top 10: ', top10
for i in xrange(10):
    #print feature_mapping[top10[i]]
    words.append(feature_mapping[top10[i]])
    
words.sort()
print '-->', words
#for i, category in enumerate(texts):
    #print 'i: ', i
    #print '############################################'
    #print 'category: ', category
    #print model.coef_[i]
    #top10 = np.argsort(model.coef_[i])[-10:]
    #print("%s: %s" % (category, " ".join(feature_names[top10])))