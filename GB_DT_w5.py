import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


data = pd.read_csv('D:\\Dev\\ML\\datasets\\coursera\\gbm-data.csv')
learning_rates = np.array([1, 0.5, 0.3, 0.2, 0.1])
X = data.iloc[:,1:].values
y = data.iloc[:,0].values


X_train, X_test, y_train, y_test = \
    train_test_split(X,y, test_size=0.8, random_state=241)

def GSB_iteration():
    learning_rates = np.array([1, 0.5, 0.3, 0.2, 0.1])
    for i in learning_rates:
        staged_decision_function(GBS(i),i)
        # staged_predict_p(GBS(i),i)
        # predict_proba(GBS(i),i)

def sigmoid(predict_y):
   return 1 / (1+np.exp(-predict_y))

def RF(n_est):
    clf = RandomForestClassifier(n_estimators=n_est, random_state=241)
    clf.fit(X_train, y_train)
    print(round(log_loss(y_test,clf.predict_proba(X_test)),2))

def GBS(learning_rate):
    clf = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=250, random_state=241)
    clf.fit(X_train, y_train)
    return clf

def staged_predict_p(clf,rate):
    test_loss = []
    train_los = []
    for p in clf.staged_predict_proba(X_test):
        test_loss.append(log_loss(y_test, sigmoid(p)))
    for p in clf.staged_predict_proba(X_train):
        train_los.append(log_loss(y_train, sigmoid(p)))
    plot(train_los, test_loss, "SPP", rate)

def staged_decision_function(clf, rate):
    test_loss = []
    train_los = []

    for p in clf.staged_decision_function(X_test):
        test_loss.append(log_loss(y_test, sigmoid(p)))

    if rate == 0.2:
        print('min_test_loss '+ str(np.argmin(test_loss)) +' '+str(round(test_loss[np.argmin(test_loss)],2)))

    for p in clf.staged_decision_function(X_train):
        train_los.append(log_loss(y_train, sigmoid(p)))

    plot(train_los, test_loss, "SDF", rate)

def predict_proba(clf,rate):
    train_loss = log_loss(y_train,clf.predict_proba(X_train))
    test_loss  = log_loss(y_test,clf.predict_proba(X_test))

    plot(train_loss,test_loss, "PP", rate)

def gen_minlog_los(loss):
    return np.argmin(loss)


def plot(train_loss, test_loss, metric, rate):
    plt.figure()
    plt.plot()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.title(metric+' learning_rate: '+str(rate))
    plt.show()


GSB_iteration()
RF(36)