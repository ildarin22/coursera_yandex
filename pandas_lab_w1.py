import pandas as pd
import re

df =  pd.read_csv('D:\\Dev\\ML\\datasets\\coursera\\titanic_data.csv', index_col='PassengerId')

df['Sex'].value_counts()                        # 1. Какое количество мужчин и женщин ехало на корабле?
df['Survived'].mean()                           # 2. Какой части пассажиров удалось выжить?
df['Pclass'].map({1:1,2:0,3:0}).mean()          # 3. Какую долю пассажиры первого класса составляли среди всех пассажиров?
df['Age'].mean()                                # 4. Какого возраста были пассажиры? Посчитайте среднее и медиану возраста
df['Age'].median()                              # два варианта: median() и mean(), либо через describe()
df['SibSp'].corr(df['Parch'],method='pearson')  # 5. Коррелируют ли число братьев/сестер/супругов с числом родителей/детей? Посчитайте корреляцию Пирсона между признаками SibSp и Parch.

df_fnames = df[df['Sex']=='female']['Name'].map(lambda x: re.sub(r'\W+', ' ', x))
df_fnames = df_fnames.str.split(' ', expand=True)
df_fnames = df_fnames.iloc[:,2:7]
df_fnames.groupby([5]).size().reset_index(name='count').sort_values(['count'])