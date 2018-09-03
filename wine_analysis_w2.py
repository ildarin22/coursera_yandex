import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

k_list = [x for x in range(1,50)]
wine_data = pd.read_csv('D:\\Dev\\ML\\datasets\\coursera\\wine.data', header=None)
wine_data_y = wine_data.iloc[1:,0]
wine_data_X = wine_data.iloc[1:,1:]
wine_data_scaled_X = scale(wine_data_X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def model_test(kf,X,y):
    scores = []
    for k in k_list:
        neigh = KNeighborsClassifier(n_neighbors=k)
        scores.append(cross_val_score(neigh, X, y, cv=kf, scoring='accuracy'))
    return pd.DataFrame(scores,k_list).mean(axis=1).sort_values(ascending=False)

print("MODEL ACCURACY {}".format(pd.DataFrame(model_test(kf,wine_data_X,wine_data_y)).head(1)))
print("MODEL ACCURACY (SCALED) {}".format(pd.DataFrame(model_test(kf,wine_data_scaled_X,wine_data_y)).head(1)))

