import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler


clf = Perceptron(random_state=241)
scaler = StandardScaler()
data_test = pd.read_csv('perceptron_test.csv', sep=',')
data_train = pd.read_csv('perceptron_train.csv', sep=',')

data_test.columns = ['class'] + ['A' + str(i) for i in range(1, 3)]
data_train.columns = ['class'] + ['A' + str(i) for i in range(1, 3)]

X_train = data_train.drop(('class'), axis=1)
y_train = data_train['class']

X_test = data_test.drop(('class'), axis=1)  # drop column 'class'
print('train: ', X_train.shape)
print('test: ', X_test.shape)
print('y train: ', y_train.shape)
y_test = data_test['class']
tempX = X_test['A1']

test_data = pd.concat((X_test['A1'], X_test['A2']))
print('test data: ', test_data.shape)

# Obrahunky
clf.fit(X_train, y_train)
pred = clf.predict(X_train)
print('==>', pred.shape)
