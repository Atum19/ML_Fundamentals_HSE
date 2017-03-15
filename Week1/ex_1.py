import pandas
from scipy.stats.stats import pearsonr

surv = 0
male = 0
p_clas = 0
female = 0

data = pandas.read_csv('titanic.csv', sep=',')
age_data = data['Age']
age_data = age_data.dropna(axis=0)
print(age_data.describe())
print('#######################################')

name = data.groupby('Name')
for line in data['Sex']:
    if line == 'male':
        male += 1
    elif line == 'female':
        female += 1

for lin in data['Survived']:
    if lin == 1:
        surv += 1

for ln in data['Pclass']:
    if ln == 1:
        p_clas += 1
print('male: ', male, 'female: ', female)

sm = male + female
print('Survived, percents ', (float(surv) / sm) * 100)
print('first clas: ', (float(p_clas) / sm) * 100)

result = pearsonr(data['SibSp'], data['Parch'])
print('correlation coeff r= %0.3f, level of significance p = %0.3f.' % result)
