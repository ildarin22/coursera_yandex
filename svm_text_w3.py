import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
    )

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data, newsgroups.target)

feature_mapping = vectorizer.get_feature_names()

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf,grid,scoring='accuracy', cv=cv)
gs.fit(X, newsgroups.target)

clf_best = gs.best_estimator_
clf_best.fit(X, newsgroups.target)

word_indexes = np.argsort(np.abs(clf_best.coef_.toarray()[0]))[-10:]
words = [feature_mapping[i] for i in word_indexes]
print(sorted(words))

# df_coef = pd.DataFrame(clf_best.coef_.data, columns = ['coef']).abs()
# df_coef = df_coef.sort_values(by=['coef'],  ascending=False)
# coef_index = df_coef['coef'].index
# words = []
# for i in range(0,10):
#     words.append(feature_mapping[coef_index[i]])
# print(sorted(words))