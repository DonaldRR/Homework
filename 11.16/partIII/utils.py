# %matplotlib inline
# import matplotlib
# matplotlib.use('TkAgg')
# from matplotlib import pyplot as plt
import cv2
from urllib import request
import numpy as np
from os import path
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

def parseUrl2Img(url):

    resp = request.urlopen(url)
    image = resp.read()
    img_arr = np.array(bytearray(image))
    img = cv2.imdecode(img_arr, -1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def colorQuan(img, n_clusters):
    shape_ = img.shape

    img = np.reshape(newshape=(-1, 3), a=img)
    print('## Fitting ...')
    kmeans = KMeans(n_clusters=n_clusters).fit(img)
    img_cluster = kmeans.predict(img)

    centroids = np.array(kmeans.cluster_centers_, dtype=int)
    new_img = np.zeros((img_cluster.shape[0], 3), dtype=int)

    for p in tqdm(range(new_img.shape[0])):
        new_img[p] = centroids[img_cluster[p]]

    return np.reshape(new_img, shape_)


def pcaCompress(img, n_components):
    shape = img.shape
    new_shape = (shape[0], shape[1] * shape[2])
    img = np.reshape(a=img, newshape=new_shape)

    if n_components > 0:
        n_components = int(n_components)

    pca = PCA(n_components=n_components)
    compressed_img = pca.fit_transform(img)

    return compressed_img, pca, shape


def pcaDecompress(img, pca, shape):
    decompressed_img = np.reshape(a=pca.inverse_transform(img), newshape=shape)
    decompressed_img = np.array(decompressed_img, dtype=int)

    return decompressed_img