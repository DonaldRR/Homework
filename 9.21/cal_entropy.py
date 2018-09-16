import numpy as np
from DecisionTree import class_counts, entropy

def new_entropy(rows, attribute_idx):
        positives = []
        negatives = []
        len_rows = float(len(rows))

        for row in rows:
                if row[attribute_idx] == 1:
                        positives.append(row)
                else:
                        negatives.append(row)

        return positives, negatives, len(positives)/len_rows * entropy(positives) + len(negatives)/len_rows * entropy(negatives)

data = [
        [0,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,1,1,0],
        [1,0,0,0],
        [1,0,1,0],
        [1,1,0,0],
        [1,1,1,0]
]
print(new_entropy(data, 2))