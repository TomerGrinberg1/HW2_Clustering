import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data

np.random.seed(2)


def add_noise(data):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :return: data + noise, where noise~N(0,0.01^2)
    """
    noise = np.random.normal(loc=0, scale=0.01, size=data.shape)
    return data + noise


def choose_initial_centroids(data, k):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :param k: number of clusters
    :return: numpy array of k random items from dataset
    """
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]


# ====================
def transform_data(df, features):
    """numbers = [1,2,3]
    Performs the following transformations on df:
        - selecting relevant features
        - scaling
        - adding noise
    :param df: dataframe as was read from the original csv.
    :param features: list of 2 features from the dataframe
    :return: transformed data as numpy array of shape (n, 2)
    """
    df = df.loc[:, features]
    col_sum = [sum(df[features[0]]), sum(df[features[1]])]
    col_min = [min(df[features[0]]), min(df[features[1]])]
    for i, col in enumerate(features):
        df[col] = df[col].apply(lambda x: (x - col_min[i]) / col_sum[i])
    df_np = add_noise(df.to_numpy())
    return df_np


def kmeans(data, k):
    """
    Running kmeans clustering algorithm.
    :param data: numpy array of shape (n, 2)
    :param k: desired number of cluster
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """
    prev_centroids = choose_initial_centroids(data, k)
    labels = assign_to_clusters(data, prev_centroids)
    flag = 0

    while flag == 0:
        labels = assign_to_clusters(data, prev_centroids)
        new_centroids = recompute_centroids(data, labels, k)
        flag = np.array_equal(prev_centroids, new_centroids)
        prev_centroids = new_centroids

    print(np.array_str(prev_centroids, precision=3, suppress_small=True))
    return labels, prev_centroids


def dist(x, y):
    """
    Euclidean distance between vectors x, y
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the euclidean distance
    """
    pass
    # return distance


def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """
    distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    labels = np.argmin(distances, axis=0)
    return labels


def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    return np.array([data[labels == k].mean(axis=0) for k in range(k)])


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """
    #all_features = list(data.columns)
    #ind1 = all_features.index('cnt')
    #ind2 = all_features.index('hum')
    #plt.scatter(data[all_features[ind1]], data[all_features[ind2]])
    colors = np.array(['cornflowerblue', 'cyan', 'magenta','salmon','palegreen'])
    plt.scatter(data[:, 0], data[:, 1], c=colors[labels])
    plt.scatter(centroids[:, 0], centroids[:, 1], c='white', edgecolors='black',
                marker='*', linewidth=2, s=100)
    plt.xlabel('cnt')
    plt.ylabel('hum')
    plt.title(f'Results for kmeans with k = {centroids.shape[0]}')
    plt.show()
    plt.savefig(path)
