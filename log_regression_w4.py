import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

salary_train = pd.read_csv("D:\\Dev\\ML\\datasets\\coursera\\salary-train.csv", names=['FullDescription','LocationNormalized', 'ContractTime','SalaryNormalized'])
salary_test = pd.read_csv("D:\\Dev\\ML\\datasets\\coursera\\salary-test-mini.csv", names=['FullDescription','LocationNormalized', 'ContractTime','SalaryNormalized'])
salary_test =  salary_test.iloc[1:]
salary_train = salary_train.iloc[1:]

salary_train['FullDescription'] = salary_train['FullDescription'].str.lower()
salary_train['FullDescription'] = salary_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
salary_train['LocationNormalized'].fillna('nan', inplace=True)
salary_train['ContractTime'].fillna('nan', inplace=True)

y_salary_train = salary_train['SalaryNormalized']

enc = DictVectorizer()
X_salary_train_cat = enc.fit_transform(salary_train[['LocationNormalized','ContractTime']].to_dict('records'))
X_salary_test_cat = enc.transform(salary_test[['LocationNormalized','ContractTime']].to_dict('records'))


vectorizer = TfidfVectorizer(min_df=5)
X_vector = vectorizer.fit_transform(salary_train['FullDescription'])
X = hstack([X_vector, X_salary_train_cat])

X_test_vector = vectorizer.transform(salary_test['FullDescription'])
X_test = hstack([X_test_vector, X_salary_test_cat])

rsn = Ridge(alpha=1, random_state=241)
rsn.fit(X, y_salary_train)
rsn.predict(X_salary_test_cat)


