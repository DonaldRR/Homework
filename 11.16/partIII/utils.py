# %matplotlib inline
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import cv2
import urllib
import numpy as np
from os import path
from tqdm import tqdm
from sklearn.cluster import KMeans

def parseUrl2Img(url):

    resp = urllib.request.urlopen(url)
    image = resp.read()
    img_arr = np.array(bytearray(image))
    img = cv2.imdecode(img_arr, -1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def colorQuan(img, n_clusters):
    shape_ = img.shape

    img = np.reshape(newshape=(-1, 3), a=img)
    kmeans = KMeans(n_clusters=n_clusters).fit(img)
    img_cluster = kmeans.predict(img)

    centroids = np.array(kmeans.cluster_centers_, dtype=int)
    new_img = np.zeros((img_cluster.shape[0], 3), dtype=int)

    for p in range(new_img.shape[0]):
        new_img[p] = centroids[img_cluster[p]]

    return np.reshape(new_img, shape_)