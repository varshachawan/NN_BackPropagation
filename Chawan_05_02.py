# Chawan, Varsha Rani
# 1001-553-524
# 2018-11-25
# Assignment-05-02

import sklearn.datasets
from sklearn.cluster import AgglomerativeClustering
import numpy as np
def generate_data(dataset_name, n_samples, n_classes):
    #########################################################################
    #  Generate the input samples of different type
    #########################################################################
    if dataset_name == 'swiss_roll':
        data1 = sklearn.datasets.make_swiss_roll(n_samples, noise=1.5, random_state=99)[0]
        data1 = data1[:, [0, 2]]
    if dataset_name == 'moons':
        data1 = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.15)[0]
    if dataset_name == 'blobs':
        data1 = sklearn.datasets.make_blobs(n_samples=n_samples, centers=n_classes*2, n_features=2, cluster_std=0.85*np.sqrt(n_classes), random_state=100)
        return data1[0]/10., [i % n_classes for i in data1[1]]
    if dataset_name == 's_curve':
        data1 = sklearn.datasets.make_s_curve(n_samples=n_samples, noise=0.15, random_state=100)[0]
        data1= data1[:, [0,2]]/3.0

    ward = AgglomerativeClustering(n_clusters=n_classes*2, linkage='ward').fit(data1)
    return data1[:]+np.random.randn(*data1.shape)*0.03, [i % n_classes for i in ward.labels_]