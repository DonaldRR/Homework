from sklearn.neural_network import MLPClassifier
import numpy as np

X = [[0.,0.], [0.,1.], [1., 0.], [1., 1.]]
y = [0, 1, 1, 0]

run_times = 1000
acc_num = 0
for i in range(run_times):
    clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(2), activation='logistic', alpha=0.002, random_state=2)
    clf.fit(X, y)
    if np.array_equal(clf.predict(X), [0, 1, 1, 0]):
        acc_num += 1

print(acc_num / run_times)