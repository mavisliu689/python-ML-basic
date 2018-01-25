import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

# For reproducibility
np.random.seed(1000)

nb_samples = 3600


def plot_clustered_dataset(X, Y):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    markers = ['o', 'd', '^', 'x', '1', '2', '3', 's']
    colors = ['r', 'b', 'g', 'c', 'm', 'k', 'y', '#cccfff']

    for i in range(nb_samples):
        ax.scatter(X[i, 0], X[i, 1], marker=markers[Y[i]], color=colors[Y[i]])

    plt.show()


if __name__ == '__main__':
    # 建立高斯分佈資料集斑點
    X, _ = make_blobs(n_samples=nb_samples, n_features=2, centers=8, cluster_std=2.0)

    # 顯示資料
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.scatter(X[:, 0], X[:, 1], marker='o', color='b')
    plt.show()

    # 完成聚合連接
    print('Complete linkage')
    ac = AgglomerativeClustering(n_clusters=8, linkage='complete')
    Y = ac.fit_predict(X)

    # 顯示群聚資料集
    plot_clustered_dataset(X, Y)

    # 平均連接
    print('Average linkage')
    ac = AgglomerativeClustering(n_clusters=8, linkage='average')
    Y = ac.fit_predict(X)

     # 顯示群聚資料集
    plot_clustered_dataset(X, Y)

    # 沃德連接
    print('Ward linkage')
    ac = AgglomerativeClustering(n_clusters=8)
    Y = ac.fit_predict(X)

    # 顯示群聚資料集
    plot_clustered_dataset(X, Y)


