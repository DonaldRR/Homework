from DecisionTree import *
import pandas as pd
from sklearn import model_selection

header = np.load('NR_names.npy')
lst = np.load('NR_data.npy')
# header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
# lst = df.values.tolist()
print('\n\n********** Building Tree ...... ***********')
t = build_tree(lst, header)
print('\n\n********** Decision Tree **********')
print_tree(t)

print("\n\n********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("\n\n********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("\n\n************* Tree before pruning *******")
print_tree(t)
acc = computeAccuracy(test, t)
print("\n\n  Accuracy on test = " + str(acc))

## TODO: You have to decide on a pruning strategy
# Pruning Strategy
print("\n\n************* Start pruning *******")
idsList = np.sort(np.array([node.id for node in innerNodes]))[::-1]
initAcc = computeAccuracy(test, t)
print('Initial accuracy:{}'.format(initAcc))

for id in idsList:
    t_pruned = copy.deepcopy(t)
    tempAcc = computeAccuracy(test, prune_tree(t_pruned, [id]))
    print('Test pruning id:{} -- accuracy:{}'.format(id, tempAcc))

    if tempAcc >= initAcc:
        print('    Prune {} Success'.format(id))
        initAcc = tempAcc
        t = t_pruned
    else:
        print('    Prune {} Fail'.format(id))
# t_pruned = prune_tree(t, [26, 11, 5])

print("\n\n*************Tree after pruning*******")
print_tree(t)
acc = computeAccuracy(test, t)
print("\n\n  Accuracy on test = " + str(acc))
