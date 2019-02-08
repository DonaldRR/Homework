import numpy as np
from tabulate import tabulate
import pandas as pd

model_names = [
    'Support Vector Machine',
    'Decision Tree',
    'Multipler-Layer Perceptrons ',
    'Gausian Naive Bayes',
    'Logistic Regression',
    'K-Neighbors',
    'Bagging',
    'Random Forest',
    'AdaBoost',
    'Gradient Boosting',
    'XGBoosting'
]

infos = []
for name in model_names:
    row = np.load('{}_result.npy'.format(name))

    alg = row[0]
    params = ''
    count_ = 0
    n_params = len(row[1])
    for k, v in row[1].items():
        count_ += 1
        params += '{}: {}'.format(k,v)
        if count_ % 3 == 0:
            params += '\n'
        if count_ % 3 != 0:
            params += ',  '
    acc = row[2]
    f1 = row[3]

    infos.append([alg, params, acc, f1])

df = pd.DataFrame(infos, columns=['Algorithm', 'Best Parameters', 'Accuracy', 'F1'])
df.to_csv('outputs/output.csv')

print(tabulate(infos, headers=['Algorithm', 'Best Parameters', 'Accuracy', 'F1'], tablefmt='grid'))