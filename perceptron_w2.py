import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

def perceptron_fit_predict(X,y,X_predict):
    clf = Perceptron(random_state=241)
    clf.fit(X,y)
    return clf.predict(X_predict)

def accuracy_printer(y_true,y_pred,scaled=False):
    print("Accuracy {} Scaled {}".format(accuracy_score(y_true,y_pred),scaled))


scaler = StandardScaler()

X_train = pd.read_csv('D:\\Dev\\ML\\datasets\\coursera\\perceptron-train.csv', header=None)
y_train = X_train.pop(0)
X_train_scaled = scaler.fit_transform(X_train)

X_test = pd.read_csv('D:\\Dev\\ML\\datasets\\coursera\\perceptron-test.csv', header=None)
y_test = X_test.pop(0)
X_test_scaled = scaler.transform(X_test)

y_pred = perceptron_fit_predict(X_train,y_train,X_test)
y_pred_scaled = perceptron_fit_predict(X_train_scaled,y_train,X_test_scaled)

accuracy_printer(y_test,y_pred)
accuracy_printer(y_test,y_pred_scaled,True)


# print("Accuracy scaled {}".format(accuracy_score(y_test,X_predict_scaled)))
