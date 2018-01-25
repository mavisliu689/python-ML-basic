#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 16:22:03 2017

@author: justinwu
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

np.random.seed(1000)

if __name__ == '__main__':
    # MNIST手寫辨識數字
    digits = load_digits()
    selection = np.random.randint(0, 1500, size=100)
    fig, ax = plt.subplots(10, 10, figsize=(10, 10))
    samples = [digits.data[x].reshape((8, 8)) for x in selection]
    for i in range(10):
        for j in range(10):
            ax[i, j].set_axis_off()
            ax[i, j].imshow(samples[(i * 8) + j], cmap='gray')
    plt.show()
    pca = PCA(n_components=36, whiten=True)
    X_pca = pca.fit_transform(digits.data / 255)
    print('解釋變異數比')
    print(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_xlabel('Component')
    ax[0].set_ylabel('Variance ratio (%)')
    ax[0].bar(np.arange(36), pca.explained_variance_ratio_ * 100.0)
    ax[1].set_xlabel('Component')
    ax[1].set_ylabel('Cumulative variance (%)')
    ax[1].bar(np.arange(36), np.cumsum(pca.explained_variance_)[::-1])
    plt.show()
    fig, ax = plt.subplots(10, 10, figsize=(10, 10))

    samples = [pca.inverse_transform(X_pca[x]).reshape((8, 8)) for x in selection]

    for i in range(10):
        for j in range(10):
            ax[i, j].set_axis_off()
            ax[i, j].imshow(samples[(i * 8) + j], cmap='gray')

    plt.show()

