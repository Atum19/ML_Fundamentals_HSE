import pandas as pd
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, make_scorer

data = pd.read_csv('abalone.csv', sep=',')

data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

y = data['Rings']
X = data.drop('Rings', axis=1)


kfold = KFold(len(X), n_folds=5, shuffle=True, random_state=1)
ftwo_scorer = make_scorer(r2_score)

for num in range(60):
    estm = num + 1
    clf = RandomForestRegressor(n_estimators=estm, random_state=1)
    clf.fit(X, y)
    predictions = clf.predict(X)
    result_m = cross_validation.cross_val_score(clf, X, y, scoring=ftwo_scorer, cv=kfold).mean()
    print('==> ', result_m, 'n: ', estm)
