import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
clf = Perceptron(random_state=241)

data_test = pd.read_csv('perceptron_test.csv', sep=',')
data_train = pd.read_csv('perceptron_train.csv', sep=',')

data_test.columns = ['class'] + ['A' + str(i) for i in range(1, 3)]
data_train.columns = ['class'] + ['A' + str(i) for i in range(1, 3)]

y_train = data_train['class']
print('y train: ', y_train)
X_train = data_train.drop(('class'), axis=1)

ser = pd.Series([])

y_test = data_test['class']
print(y_test.shape)
X_test = data_test.drop(('class'), axis=1)  # drop column 'class'

X_train['A1'] = X_train['A1'] * 1000
X_test['A1'] = X_test['A1'] * 1000

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print('==>', pred.shape)
