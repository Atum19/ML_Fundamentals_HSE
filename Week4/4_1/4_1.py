import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


d_vectorizer = DictVectorizer()
t_vectorizer = TfidfVectorizer(min_df=5)
model = Ridge(alpha=1, random_state=241)

data_train = pd.read_csv('salary-train.csv', sep=',')
data_test = pd.read_csv('salary-test-mini.csv', sep=',')

data_train['FullDescription'] = [elem.lower() for elem in
                                 data_train['FullDescription'][0:]]

data_train['FullDescription'] = data_train['FullDescription'].\
                                replace('[^a-zA-Z0-9]', ' ', regex=True)

data_test['FullDescription'] = [elem.lower() for elem in
                                data_test['FullDescription'][0:]]

data_test['FullDescription'] = data_test['FullDescription'].\
                                replace('[^a-zA-Z0-9]', ' ', regex=True)

text_train = t_vectorizer.fit_transform(data_train['FullDescription'])
text_test = t_vectorizer.transform(data_test['FullDescription'])

data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)
data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

X_train_categ = d_vectorizer.fit_transform(data_train[['LocationNormalized',
                                                       'ContractTime']].to_dict('records'))
X_test_categ = d_vectorizer.transform(data_test[['LocationNormalized',
                                                 'ContractTime']].to_dict('records'))
                                
combined_train = hstack([X_train_categ, text_train])
model.fit(combined_train, data_train['SalaryNormalized'])
combined_test = hstack([X_test_categ, text_test])
pred_vals = model.predict(combined_test)
print('==>', pred_vals)
