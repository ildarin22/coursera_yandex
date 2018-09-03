import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score

data = pd.read_csv('D:\\Dev\\ML\\datasets\\coursera\\abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else(-1 if x == 'F' else 0))
scores = []

y = data.iloc[:,8]
X = data.iloc[:,0:8]

kf = KFold(n_splits=5,shuffle=True,random_state=241)

for i in range(1,51):
    rfr = RandomForestRegressor(n_estimators=i, random_state=1)
    rfr.fit(X,y)
    y_pred = rfr.predict(X)
    print(i,np.round(np.mean(cross_val_score(rfr, X, y, cv=kf)),2), r2_score(y,y_pred))