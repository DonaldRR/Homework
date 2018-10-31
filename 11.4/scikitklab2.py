# This example has been taken from SciKit documentation and has been
# modifified to suit this assignment. You are free to make changes, but you
# need to perform the task asked in the lab assignment


from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
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
digits = datasets.load_digits()


# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts into 80:20 ratio for train:test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# This is a key step where you define the parameters and their possible values
# that you would like to check.


tuned_parameters = {
    # 'svc':[SVC(),
    #        [{'degree':[2, 3, 4, 5],
    #          'kernel': ['rbf', 'linear', 'poly'],
    #         'gamma': [1e-3, 1e-4, 1e-5],
    #         'C': [1, 10, 100, 1000]}]],
    # 'dt':[DecisionTreeClassifier(),
    #       [{'criterion':['gini', 'entropy'],
    #         'max_depth':[8, 12, 16, 20],
    #         'min_samples_split':[2, 5, 10],
    #         'min_samples_leaf':[1, 2, 4, 8],
    #         'min_weight_fraction_leaf':[0., 0.01, 0.05],
    #         'max_features':[None, 'sqrt', 'log2'],
    #         'min_impurity_decrease':[0., 0.02, 0.05]}]],
    # 'nn':[MLPClassifier(),
    #       [{'hidden_layer_sizes':[(16,16),(32,16),(16,16,8)],
    #         'activation':['identity', 'relu','logistic'],
    #         'alpha':[0.0001, 0.0005, 0.001],
    #         'learning_rate':['adaptive', 'invscaling'],
    #         'max_iter':[200, 500, 1000]}]],
    # 'gNB':[GaussianNB(),
    #        [{'priors':[None, [0.1]*10]}]],
    # 'logit':[LogisticRegression(),
    #          [{'penalty':['l1','l2'],
    #            'tol':[1e-4, 1e-5, 1e-6],
    #            'C':[0.5, 1.0, 1.5],
    #            'fit_intercept':[True, False],
    #            'max_iter':[100, 200, 500]}]],
    # 'knn':[KNeighborsClassifier(),
    #        [{'n_neighbors':[5, 10, 15],
    #          'weights':['uniform', 'distance'],
    #          'algorithm':['ball_tree', 'kd_tree', 'brute'],
    #          'p':[1, 2]}]],
    # 'bagging':[BaggingClassifier(),
    #            [{'n_estimators':[10, 15, 20],
    #              'max_samples':[0.2, 0.5, 0.8],
    #              'max_features':[1, 2, 3],
    #              'random_state':[1, 2, 3]}]],
    # 'rf':[RandomForestClassifier(),
    #       [{'n_estimators':[10, 15, 20],
    #         'criterion':['gini', 'entropy'],
    #         'max_depth':[None, 10, 15],
    #         'max_features':[None, 'sqrt', 'log2'],
    #         'min_samples_split':[2, 5, 10]}]],
    # 'adaboost':[AdaBoostClassifier(),
    #             [{'n_estimators':[50, 80, 100],
    #               'learning_rate':[1., 0.5],
    #               'algorithm':['SAMME', 'SAMME.R'],
    #               'random_state':[1, 2, 3]}]],
    # 'gb':[GradientBoostingClassifier(),
    #       [{'loss':['deviance'],#, 'exponential'],
    #         'learning_rate':[0.1],#, 0.05, 0.01],
    #         'n_estimators':[100],# 200, 300],
    #         'max_depth':[2],# 3, 4],
    #         'min_samples_split':[2]}]],#, 5, 8]}]],
    'xgb':[XGBClassifier(),
           [{'learning_rate':[0.1, 0.05, 0.02],
             'n_estimators':[200, 300, 400],
             'min_child_weight':[3, 4, 5],
             'booster':['gbtree', 'gblinear', 'dart']}]]
}

# We are going to limit ourselves to accuracy score, other options can be
# seen here:
# http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# Some other values used are the predcision_macro, recall_macro
scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    for k, v in tuned_parameters.items():
        print("============== Current Model: {} ================".format(k))

        clf = GridSearchCV(v[0], v[1], cv=5, scoring='%s' % score)

        clf.fit(X_train, y_train)

        print("\tBest parameters set found on development set:")
        print()
        print("\t",clf.best_params_)
        print()
        print("\tGrid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("\t%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("\tDetailed classification report:")
        print()
        print("\tThe model is trained on the full development set.")
        print("\tThe scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print("\t",classification_report(y_true, y_pred))
        print("\tDetailed confusion matrix:")
        print("\t",confusion_matrix(y_true, y_pred))
        print("\tAccuracy Score: \n")
        print("\t",accuracy_score(y_true, y_pred))

        print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.
