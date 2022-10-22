import sklearn.model_selection
import sklearn.feature_extraction
import pandas as pd
import random
import numpy as np

random.seed(0)

train_data = pd.read_csv('processed_train.csv')
# train_data = train_data.drop(['PassengerId', 'Ticket', 'Name', 'Cabin'], axis=1)
survived = train_data['Survived']
train_data = train_data.drop('Survived', axis=1)
# print( train_data['Embarked'].mode() )

# for col in ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']:
# #    train_data[col] = train_data[col].fillna(train_data[col].mode())
#     train_data.loc[train_data[col].isnull(), col] = train_data[col].mode().values

# for col in ['Age', 'Fare']:
# #    train_data[col] = train_data[col].fillna(train_data[col].mean())
#     train_data.loc[train_data[col].isnull(), col] = train_data[col].mean()
    
print(train_data.isnull().sum().sort_values(ascending=False))

# DictVectorizaion takes features that are strings, and converts them to booleans where each string becomes a column that is a feature
# if there are multiple of them, then it will count the frequency (still gives depth??)
vectorizer = sklearn.feature_extraction.DictVectorizer(sparse=False)
#print([elem for elem in vectorizer.fit_transform(train_data.to_dict(orient='records'))[0, :]])
train_data =  np.hstack( (vectorizer.fit_transform(train_data.to_dict(orient='records')), survived.values.reshape((-1,1 ) ) ) )
#train_data['Survived'] = survived

print("Shape of train:", train_data.shape)

#train_data = train_data[[c for c in train_data if c not in ['Survived']] + ['Survived']]
kf = sklearn.model_selection.KFold(n_splits=5)

for i, (train_index, test_index) in enumerate(kf.split(train_data)):
    #train_data.iloc[train_index, :].to_csv('train_' + str(i) + '.csv.gz', compression='gzip', index=False)
    #train_data.iloc[test_index, :].to_csv('test_' + str(i) + '.csv.gz', compression='gzip', index=False)
    
    np.savetxt('train_' + str(i) + '.csv.gz', train_data[train_index], delimiter=',')
    np.savetxt('test_' + str(i) + '.csv.gz', train_data[test_index], delimiter=',')

