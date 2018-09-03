from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np

X = pd.read_csv('D:\\Dev\\ML\\datasets\\coursera\\svm-data.csv', header=None)
y = X.pop(0)

clf = SVC(C=10000,kernel='linear', random_state=241)
clf.fit(X,y)

decision = clf.decision_function(X)
plot_decision_regions(X=X.values, y=np.array(y.values,dtype='int64'), clf=clf, legend=2)

plt.xlabel(X.columns[0], size=14)
plt.ylabel(X.columns[1], size=14)
plt.title('SVM Decision Region Boundary', size=16)
plt.show()