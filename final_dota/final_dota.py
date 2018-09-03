import pandas as pd
import numpy as np
import datetime

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, validation_curve, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt


# Init train data
def train_data_init(csv_path):
    data = pd.read_csv(csv_path, index_col='match_id').fillna(0)
    X = data.ix[:, : 'dire_first_ward_time']
    y = data.pop('radiant_win')
    return X, y


# One Hot Encoding

def words_bag(data, N):
    X_pick = np.zeros((data.shape[0], N))

    for i, match_id in enumerate(data.index):
        for p in range(5):
            X_pick[i, data.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
            X_pick[i, data.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
    return X_pick



# Searching best params for Logistic Regression and Gradient Boosting
def best_param_search(X, y, clf, param_range, kf, model):
    if model is 'gbc':
        grid = {'n_estimators': param_range}
        gs = GridSearchCV(clf, grid , scoring='roc_auc', cv=kf)
        gs.fit(X, y)
        print('GBC Max auc_roc:', gs.best_score_)
        return gs.best_estimator_

    if model is 'lr':
        lr = LogisticRegressionCV(param_range, cv=kf, scoring='roc_auc')
        lr.fit(X, y)
        print('LR Max auc_roc:', lr.scores_[1].max())
        return lr.C_


# Plotting Models Curves
def curve_plotting(train_scores, test_scores, param_name=None, title=None):
    plt.plot(train_scores, color='r', label='train')
    plt.plot(test_scores, color='g', label='test')
    plt.xlabel(param_name)
    plt.ylabel('AUC-ROC')
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()


def lr_val_curve(clf, kf, X, y, param_name, param_range):
    train_scores, test_scores = validation_curve(estimator=clf, X=X, y=y, param_name=param_name,
                                                 param_range=param_range,
                                                 cv=kf,
                                                 scoring='roc_auc')

    train_scores_mean = np.mean(train_scores, axis=1).round(3)
    test_scores_mean = np.mean(test_scores, axis=1).round(3)

    return train_scores_mean, test_scores_mean



def gbs_val_curve(clf, X, y):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.5, shuffle=True)
    clf.fit(X_train, y_train)
    test_sc = np.empty(len(clf.estimators_))
    train_sc = np.empty(len(clf.estimators_))

    for i, p in enumerate(clf.staged_predict_proba(X_test)):
        test_sc[i] = roc_auc_score(y_test, p[:, 1])

    for i, p in enumerate(clf.staged_predict_proba(X_train)):
        train_sc[i] = roc_auc_score(y_train, p[:, 1])

    return train_sc, test_sc

# Estimate ROC curve
def roc_score(kf, clf, X, y):
    start_time = datetime.datetime.now()
    cv = cross_val_score(clf, X, y, scoring='roc_auc', cv=kf)

    return str(np.round(np.mean(cv), 6)), str(datetime.datetime.now() - start_time)


def gbs_param_test(kf, X, y, param_range, plot):
    for i in param_range:
        clf = GradientBoostingClassifier(n_estimators=i)
        roc, end_time = roc_score(kf, clf, X, y)
        if (plot):
            gbs_plotting(clf, X, y, 'n_estimators', i)
        print("n_estimators: ", i,
              "| AUC-ROC: ", roc,
              "| Time: ", end_time)
    print('\n')


def gbs_plotting(clf, X, y, param_name, param_val):
    title = 'GBC ' + param_name + ' = ' + str(param_val)
    train_scores, test_scores = gbs_val_curve(clf, X, y)
    curve_plotting(train_scores, test_scores,
                   param_name='n_estimators', title=title)


def lr_param_test(kf, X, y, param_range, testing_type):
    for i in param_range:
        clf = LogisticRegression(C=i)
        roc, end_time = roc_score(kf, clf, X, y)
        print("Test type: ", testing_type,
              "| C: ", i,
              "| AUC-ROC: ", roc,
              "| Time: ", end_time,)
    print('\n')


def lr_data_optimizer(optimize_type, X):
    std = StandardScaler()
    X_cleaned = X.drop(['lobby_type',
                        'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                        'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'
                        ], axis=1)

    if optimize_type is 'full_data':
        return std.fit_transform(X)

    if optimize_type is 'clean_data':
        return std.fit_transform(X_cleaned)

    if optimize_type is 'extended_data':
        X_cleaned.reset_index(drop=True, inplace=True)
        X_pick = pd.DataFrame(words_bag(X, 112)).reset_index(drop=True)
        X_ext = pd.concat([X_cleaned, X_pick], axis=1)
        return std.fit_transform(X_ext)


def lr_test(kf, X, y, testing_param, param_range):
    if testing_param is 'clean_data':
        X = lr_data_optimizer(testing_param, X)
        lr_param_test(kf, X, y, param_range,testing_param)

    if testing_param is 'full_data':
        X = lr_data_optimizer(testing_param, X)
        lr_param_test(kf, X, y, param_range,testing_param)

    if testing_param is 'extended_data':
        X = lr_data_optimizer(testing_param, X)
        lr_param_test(kf, X, y, param_range,testing_param)


def gbc_test(kf, X, y, param_range, plot=False):
    gbs_param_test(kf, X, y, param_range,  plot=plot)


def gbc_get_best_param(X, y, kf, estimators):
    return best_param_search(X, y, GradientBoostingClassifier(), estimators, kf, 'gbc')


def lr_get_best_param(X, y, kf, testing_param, param_range):
    X_ext = lr_data_optimizer(testing_param, X)
    lr = best_param_search(X_ext, y, LogisticRegression(), param_range, kf, 'lr')

    return lr


def lr_data_predict(X, y, X_test, kf, param_range):
    lr_param = lr_get_best_param(X_train, y, kf, 'extended_data', param_range)
    clf = LogisticRegressionCV(lr_param, cv=kf)
    clf.fit(X, y)
    return np.min(clf.predict_proba(X_test)[:, 1]), np.max(clf.predict_proba(X_test)[:, 1])


def gbc_data_predict(X, y, X_test, kf, estimators):
    gbc_best = gbc_get_best_param(X, y, kf, estimators)
    gbc_best.fit(X,y)
    return np.min(gbc_best.predict_proba(X_test)[:, 1]), np.max(gbc_best.predict_proba(X_test)[:, 1])




train_csv_path = #path_to_train_data
test_csv_path = #path_to_test_data

X_train, y = train_data_init(train_csv_path)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_test = pd.read_csv(test_csv_path, index_col='match_id').fillna(0)
X_train_optimized = lr_data_optimizer('extended_data', X_train)
X_test_optimized = lr_data_optimizer('extended_data', X_test)

C_param_range = np.power(10.0, np.arange(-3, 4))
estimators = [30, 40, 50]




#  Logistic Regression Test, by time and AUC-ROC metrics
# 'testing_param' values: extended_data, clean_data, full_data
# 'plot' boolen - plotting

lr_test(kf, X_train, y, 'extended_data', C_param_range)
lr_test(kf, X_train, y, 'full_data', C_param_range)
lr_test(kf, X_train, y, 'clean_data',C_param_range)


# Gradient Boosting Classifier Test, by time and AUC-ROC metrics
# 'plot' boolen - plotting

gbc_test(kf, X_train, y, estimators, plot=True)



# print(lr_data_predict(X_train_optimized, y, X_test_optimized, kf,C_param_range))
# print(gbc_data_predict(X_train, y, X_test, kf))

