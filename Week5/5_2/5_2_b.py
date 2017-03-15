import pandas as pd
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


datafrm = pd.read_csv('gbm-data.csv', sep=',')

grad_boost = GradientBoostingClassifier

y_trn = datafrm['Activity']
x_trn = datafrm.drop('Activity', axis=1)
yf_trn = y_trn.values
xf_trn = x_trn.values

X_train, X_test, y_train, y_test = train_test_split(xf_trn, yf_trn,
                                                    test_size=0.8,
                                                    random_state=241)

tree_clf = RandomForestClassifier(n_estimators=8, random_state=241)
tree_clf.fit(X_train, y_train)
pred = tree_clf.predict_proba(X_test)
print(round(log_loss(y_test, pred), 2))
