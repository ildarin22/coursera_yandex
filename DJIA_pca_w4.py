import numpy as np
from sklearn.decomposition import PCA #
import pandas as pd


close_prices = pd.read_csv("D:\\Dev\\ML\\datasets\\coursera\\close_prices.csv")
djia_index = pd.read_csv("D:\\Dev\\ML\\datasets\\coursera\\djia_index.csv")
X = close_prices.iloc[:,1:31]
X_dj = djia_index.iloc[:,1]


pca = PCA(n_components=10)
pca.fit(close_prices)

print(pca.explained_variance_ratio_)  # содержит процент дисперсии, который объясняет каждая компонента
print(pca.components_)                # содержит информацию о том, какой вклад вносят признаки в компоненты