from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np


def derivative(data_, answer_, index, position, c):
    sum_value = 0
    for j in range(len(answer_)):
        sum_value += answer_[j] * data_[j][index] * (
            1 - 1 / (1 + np.exp(-answer_[j] * (
                position[0] * data_[j][0] + position[1] * data_[j][1]
            )))
        )
    sum_value /= data_.size
    return sum_value - c * position[index]


def grad(data_, answer_, position, c):
    return np.array([derivative(data_, answer_, i, position, c) for i in range(len(position))])


MAX_NUMBER_ITERATION = 10000
EPS = 1e-5


def grad_search(data_, answer_, start_position, k, c):
    position = start_position
    for i in range(MAX_NUMBER_ITERATION):
        nposition = position + grad(data_, answer_, position, c) * k
        distance = np.linalg.norm(nposition - position)
        position = nposition
        if distance < EPS:
            break
    return position


def prob(point, parameter):
    return 1 / (1 + np.exp(-parameter[0] * point[0] - parameter[1] * point[1]))


def predict(data_, parameter):
    return [prob(data_[el], parameter) for el in range(len(data_))]

def score(data_, answer_, c, k=0.1):
    parameter = grad_search(data_, answer_, np.zeros(2), k, c)
    p = predict(data_, parameter)
    return roc_auc_score(answer_, predict(data_, parameter))


input_data = pd.read_csv('D:\\Dev\\ML\\datasets\\coursera\\data-logistic.csv')
data = input_data[input_data.columns[1:]].get_values()
answer = input_data[input_data.columns[0]].get_values()

print(score(data, answer, 0), score(data, answer, 10))