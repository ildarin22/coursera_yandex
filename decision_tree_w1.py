import pandas as pd
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=241)

df_data = pd.read_csv('D:\\Dev\\ML\\datasets\\coursera\\titanic_data.csv', usecols=['Survived','Pclass','Fare','Age','Sex'])
df_data['Sex'].replace(['female', 'male'],[1,0], inplace=True)
df_x_data = df_data.dropna()
df_y_data = df_x_data['Survived']
df_x_data = df_x_data.drop(['Survived'], axis=1)

clf.fit(df_x_data,df_y_data)
clf.feature_importances_