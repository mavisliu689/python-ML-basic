#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 19:42:12 2017

@author: justinwu
"""

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt




# For reproducibility
np.random.seed(880)

nb_samples = 880


def show_dataset(X, Y):
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))

    ax.grid()
    ax.set_ylabel('Y')
    ax.set_xlabel('X')


    for i in range(nb_samples):
        if Y[i] == 0:
            ax.scatter(X[i, 0], X[i, 1], marker='o', color='r')
        else:
            ax.scatter(X[i, 0], X[i, 1], marker='^', color='b')

    plt.show()


def show_clustered_dataset(X, Y):
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))

    ax.grid()
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    

    for i in range(nb_samples):
        if Y[i] == 0:
            ax.scatter(X[i, 0], X[i, 1], marker='o', color='r')
        else:
            ax.scatter(X[i, 0], X[i, 1], marker='^', color='b')

    plt.show()


if __name__ == '__main__':
     # 建立資料集半月形資料
    X, Y = make_moons(n_samples=nb_samples, noise=0.05)

    # 顯示資料
    show_dataset(X, Y)

    # 建立和訓練密度為基礎的空間群聚DBSCAN
    #eps為定義兩個鄰居間的最大距離
    dbs = DBSCAN(eps=0.1)
    Y = dbs.fit_predict(X)

    # 顯示群聚資料集
    show_clustered_dataset(X, Y)
    #print('Cluster labels:%s' % Y)
   # for i in range(len(X[:,0])):
   #     print(X[i,0],X[i,1])
