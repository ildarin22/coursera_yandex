import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

data_class = pd.read_csv("D:\\Dev\\ML\\datasets\\coursera\\metrics_classification.csv", header=None)
data = np.array(data_class.ix[1:]).astype(int)

y_true = data[:, 0]
y_pred = data[:, 1]

data_scores = pd.read_csv("D:\\Dev\\ML\\datasets\\coursera\\metrics_scores.csv", header=None)
scores_names = data_scores.ix[0, 1:4].as_matrix()
scores = np.array(data_scores.ix[1:]).astype(float)
y_scores = scores[:, 0]
x_scores = scores[:, 1:5]


def prc_scoring():
    for i in range(x_scores.shape[1]):
        precision, recall, thresholds = precision_recall_curve(y_scores, x_scores[:,i])
        precision_7 = np.take(precision, np.where(recall > 0.7))
        print(scores_names[i], precision_7[0 , np.argmax(precision_7)])


def roc_score():
    for i in range(x_scores.shape[1]):
        print(scores_names[i], roc_auc_score(y_scores, x_scores[:,i]))


def metric_mat():
    metric_matrix = np.array([[1, 1], [0, 1], [0, 0], [1, 0]])
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(len(data)):
        if np.array_equal(data[i], metric_matrix[0]):
            TP += 1
        if np.array_equal(data[i], metric_matrix[1]):
            FP += 1
        if np.array_equal(data[i], metric_matrix[2]):
            TN += 1
        if np.array_equal(data[i], metric_matrix[3]):
            FN += 1
    print("TP: "+str(TP) , "FP: "+str(FP), "TN: "+str(TN), "FN: "+str(FN))

def clf_scores():
    print("accuracy_score: "+str(round(accuracy_score(y_true, y_pred),2)))
    print("precision_score: "+str(round(precision_score(y_true, y_pred),2)))
    print("recall_score: "+str(round(recall_score(y_true, y_pred),2)))
    print("f1_score:"+str(round(f1_score(y_true, y_pred),2)))

