import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


scaler = StandardScaler()
clf = Perceptron(random_state=241)

data_test = pd.read_csv('perceptron_test.csv', sep=',')
data_train = pd.read_csv('perceptron_train.csv', sep=',')

data_test.columns = ['class'] + ['A' + str(i) for i in range(1, 3)]
data_train.columns = ['class'] + ['A' + str(i) for i in range(1, 3)]

y_train = data_train['class']
X_train = data_train.drop('class', axis=1)

y_test = data_test['class']
X_test = data_test.drop('class', axis=1)  # drop column 'class'
tempX = X_test['A1']

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print('==>', pred.shape)
accur = accuracy_score(y_test, pred)
print('accuracy: ', accur)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf.fit(X_train_scaled, y_train)
pred = clf.predict(X_train)
pred_scal = clf.predict(X_test_scaled)
accur_scal = accuracy_score(y_test, pred_scal)
print('accuracy scaled: ', accur_scal)
print('accur diff: ', accur_scal - accur)
