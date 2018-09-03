import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor

estate_data = datasets.load_boston()
estate_data_X = estate_data['data']
estate_data_y = estate_data['target']
estate_data_labels = estate_data['feature_names']
estate_data_scaled_X = scale(estate_data_X)


kf = KFold(n_splits=5, random_state=42, shuffle=True)

def regression_model_test(kf,X,y):
    p_range = np.ndarray.tolist(np.linspace(1,10,200))
    score = []
    for p in p_range:
        rm = KNeighborsRegressor(n_neighbors=5, weights='distance',p=p, metric='minkowski')
        score.append(cross_val_score(rm,X,y,scoring='neg_mean_squared_error',cv=kf))
        print("{} {}".format(cross_val_score(rm,X,y,scoring='neg_mean_squared_error',cv=kf),p))
    return pd.DataFrame(score, p_range).mean(axis=1).sort_values(ascending=False)

# regression_model_test(kf,estate_data_scaled_X,estate_data_y)
print(pd.DataFrame(regression_model_test(kf,estate_data_scaled_X,estate_data_y,)).head(1))




