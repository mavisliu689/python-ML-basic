#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:22:28 2017

@author: justinwu
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# For reproducibility
np.random.seed(880)

nb_samples = 880


def show_dataset(X):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.grid()
    ax.set_xlabel('Salary')
    ax.set_ylabel('Age')

    ax.scatter(X[:, 0], X[:, 1], marker='o', color='b')

    plt.show()


def show_clustered_dataset(X, kmeansCluster):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.grid()
    ax.set_xlabel('Salary')
    ax.set_ylabel('Age')

    for i in range(nb_samples):
        c = kmeansCluster.predict(X[i].reshape(1, -1))
        if c == 0:
            ax.scatter(X[i, 0], X[i, 1], marker='o', color='r')
        elif c == 1:
            ax.scatter(X[i, 0], X[i, 1], marker='^', color='b')
        else:
            ax.scatter(X[i, 0], X[i, 1], marker='d', color='g')

    plt.show()


if __name__ == '__main__':
    # 建立資料集
    X, _ = make_blobs(n_samples=nb_samples, n_features=2, 
                      centers=3, cluster_std=0.5)
    # 顯示資料
    show_dataset(X)
    # 建立和顯示 K-Means
    kmCluster = KMeans(n_clusters=3)
    kmCluster.fit(X)
    # 顯示中心
    print(kmCluster.cluster_centers_)
    # 顯示群集資料集
    show_clustered_dataset(X, kmCluster)
    