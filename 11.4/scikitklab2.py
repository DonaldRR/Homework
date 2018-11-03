# This example has been taken from SciKit documentation and has been
# modifified to suit this assignment. You are free to make changes, but you
# need to perform the task asked in the lab assignment


from __future__ import print_function

import time
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost.sklearn import XGBClassifier


print(__doc__)

# Loading the Digits dataset
df = pd.read_csv('anuran_data.csv')
# digits = datasets.load_digits()


# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = df.shape[0]
X, y = df.values[:, :23], df.values[:, 23]
y = y.reshape((n_samples))
# n_samples = len(digits.images)
# X = digits.images.reshape((n_samples, -1))
# y = digits.target

# Split the dataset in two equal parts into 80:20 ratio for train:test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# This is a key step where you define the parameters and their possible values
# that you would like to check.


tuned_parameters = {
    'Support Vector Machine':[SVC(),
           [{'degree':[2, 3, 4, 5],
             'kernel': ['rbf', 'linear', 'poly'],
            'gamma': [1e-3, 1e-4, 1e-5],
            'C': [1, 10, 100, 1000]}],
           None],
    'Decision Tree':[DecisionTreeClassifier(),
          [{'criterion':['gini', 'entropy'],
            'max_depth':[8, 12, 16, 20],
            'min_samples_split':[2, 5, 10],
            'min_samples_leaf':[1, 2, 4, 8],
            'min_weight_fraction_leaf':[0., 0.01, 0.05],
            'max_features':[None, 'sqrt', 'log2'],
            'min_impurity_decrease':[0., 0.02, 0.05]}],
          None],
    'Multipler-Layer Perceptrons ':[MLPClassifier(),
          [{'hidden_layer_sizes':[(16,16),(32,16),(16,16,8)],
            'activation':['identity', 'relu','logistic'],
            'alpha':[0.0001, 0.0005, 0.001],
            'learning_rate':['adaptive', 'invscaling'],
            'max_iter':[200, 500, 1000]}],
          None],
    'Gausian Naive Bayes':[GaussianNB(),
           [{'priors':[None, [0.1]*10]}],
           None],
    'Logistic Regression':[LogisticRegression(),
             [{'penalty':['l1','l2'],
               'tol':[1e-4, 1e-5, 1e-6],
               'C':[0.5, 1.0, 1.5],
               'fit_intercept':[True, False],
               'max_iter':[100, 200, 500]}],
             None],
    'K-Neighbors':[KNeighborsClassifier(),
           [{'n_neighbors':[5, 10, 15],
             'weights':['uniform', 'distance'],
             'algorithm':['ball_tree', 'kd_tree', 'brute'],
             'p':[1, 2]}],
           None],
    'Bagging':[BaggingClassifier(),
               [{'n_estimators':[10, 15, 20],
                 'max_samples':[0.2, 0.5, 0.8],
                 'max_features':[1, 2, 3],
                 'random_state':[1, 2, 3]}],
               None],
    'Random Forest':[RandomForestClassifier(),
          [{'n_estimators':[10, 15, 20],
            'criterion':['gini', 'entropy'],
            'max_depth':[None, 10, 15],
            'max_features':[None, 'sqrt', 'log2'],
            'min_samples_split':[2, 5, 10]}],
          None],
    'AdaBoost':[AdaBoostClassifier(),
                [{'n_estimators':[50, 80, 100],
                  'learning_rate':[1., 0.5],
                  'algorithm':['SAMME', 'SAMME.R'],
                  'random_state':[1, 2, 3]}],
                None],
    'Gradient Boosting':[GradientBoostingClassifier(),
          [{'loss':['deviance', 'exponential'],
            'learning_rate':[0.1, 0.05, 0.01],
            'n_estimators':[100, 200, 300],
            'max_depth':[2, 3, 4],
            'min_samples_split':[2, 5, 8]}],
          None],
    'XGBoosting':[XGBClassifier(),
           [{'learning_rate':[0.1, 0.05, 0.02],
             'n_estimators':[200, 300, 400],
             'min_child_weight':[3, 4, 5],
             'booster':['gbtree', 'gblinear', 'dart']}],
           None]
}


scores = ['accuracy']

print("# Tuning hyper-parameters for %s" % scores)
print()

for k, v in tuned_parameters.items():
    t1 = time.time()
    print("# Current Model: {} ".format(k))

    model_name = k
    model = v[0]
    params = v[1]

    clf = GridSearchCV(v[0], v[1], cv=5, scoring=scores[0])
    clf.fit(X_train, y_train)

    tuned_parameters[model_name][2] = clf

    t2 = time.time()
    print("### Run time: {}s".format(round(t2-t1, 2)))

print()

results = []
for k, v in tuned_parameters.items():

    model_name = k
    model = v[0]
    params = v[1]
    clf = v[2]

    y_true, y_pred = y_test, clf.predict(X_test)

    algorithm = model_name
    best_params = clf.best_params_
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    results.append([algorithm, best_params, accuracy, f1_macro])

from tabulate import tabulate
print(tabulate(results, headers=['Algorithm', 'Best Parameters', 'Accuracy', 'F1(marco)']))