import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

data = np.array(pd.read_csv('D:\\Dev\\ML\\datasets\\coursera\\data-logistic.csv').as_matrix())
X = data[:,1:3]
y = data[:,0].reshape([-1,1])
k = 0.1




def sigmoid(a):
    return 1.0 - 1.0 / (1.0 + np.exp(-y*a))

def deriv(i,w1w2,C):
    d = sigmoid(np.matmul(X, w1w2))
    sum = np.sum((y * X[:, i].reshape(-1,1)) * d) / X.size
    return sum - C * w1w2[i]

def gradient(w1w2, C):
    return np.array([deriv(i, w1w2, C) for i in range(len(w1w2))])

def gradient_descent(C):
    e = 1e-5
    w1w2 = np.array([[0], [0]])

    for i in range(10000):
        w1w2_n = w1w2 + gradient(w1w2, C) * k
        dis = np.linalg.norm(w1w2_n - w1w2)
        w1w2 = w1w2_n
        if dis < e:
            break
    return w1w2

def probability(w1w2):
    return 1 / (1 + np.exp(np.matmul(-w1w2.T, X.T)))


def score(C):
    w1w2 = gradient_descent(C)
    return roc_auc_score(y, probability(w1w2).T)



if __name__ == "__main__":
    print(round(score(0),3), round(score(10),3))