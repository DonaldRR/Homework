import json
from urllib import request
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Parse Json from URL
def parseJson(url):
    json_page = request.urlopen(url)
    json_content = json_page.read().decode('utf-8')
    json_iter = re.finditer(r'"}}', json_content)
    indices = [t.start(0) for t in json_iter]

    start = 0
    jsons = []
    for i in range(len(indices)):
        end = indices[i] + 3
        jsons.append(json.loads(json_content[start:end]))
        start = end + 1

    return jsons

# Compute Jaccard Distance
def JaccardDist(str1, str2):

    if not isinstance(str1, str):
        str1 = ' '.join(str1)
    if not isinstance(str2, str):
        str2 = ' '.join(str2)

    vectorizer = CountVectorizer()

    bag = vectorizer.fit_transform([str1,str2])
    bag = bag.toarray()
    bag = np.array(np.array(bag, dtype=np.bool), dtype=int)

    return 1. - float((np.dot(bag[0], bag[1])))/(len(vectorizer.get_feature_names()))


def computeMean(a, thres):
    vectorizer = CountVectorizer()

    bag = vectorizer.fit_transform(a)
    bag = bag.toarray()
    mean_ = np.mean(bag, axis=0)
    mean_ = np.array(np.array(mean_[mean_ > thres], dtype=np.bool), dtype=int)

    return vectorizer.inverse_transform(np.array([mean_ > thres], dtype=int))