"""
The current configuration is set up to visualise the PCA results for the 2D and 3D features for a SINGLE CLASS.
For studying inter-class relationships, the code can be modified to visualise the PCA results for the 2D and 3D features for MULTIPLE CLASSES.
"""

import torch
import os
import numpy as np
import shutil
from sklearn.manifold import TSNE
import mpld3
from time import time
import argparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


import random
random.seed(404543)
torch.manual_seed(404543)

def get_original_list(folder):
    original_list = []
    for filename in list(os.listdir(folder)):
        sample = torch.load(os.path.join(folder, filename))
        if sample is not None:
            original = sample
            original = original.detach().cpu().numpy()
            original = [filename, original]
            original_list.append(original)
    return np.array(original_list, dtype=object)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualise PCA results')
    parser.add_argument('--feats_path', type=str, default='/saved/from/extract_features', help='Path where the features are saved')
    parser.add_argument('--dimension', type=str, default='2', help='2D or 3D')
    parser.add_argument('--kmeans', type=int, default=0, help='Number of images to downsample to')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    print(args)

    class_folder = args.feats_path
    original_list = get_original_list(class_folder)
    original_array = np.concatenate(original_list[:,1], axis=0)

    if args.dimension == '2':
        pca = PCA(n_components=2, svd_solver='full', random_state=404543)
        pca_results = pca.fit_transform(original_array)
    elif args.dimension == '3':
        pca = PCA(n_components=3, svd_solver='full', random_state=404543)
        pca_results = pca.fit_transform(original_array)
    else:
        print('Invalid dimension')
        exit()

    # plot the PCA results
    if args.dimension == '2':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(pca_results[:, 0], pca_results[:, 1])
        ax.set_title('PCA')
        ax.set_xlabel('1st eigenvector')
        ax.set_ylabel('2nd eigenvector')
        plt.show()
    elif args.dimension == '3':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_results[:, 0], pca_results[:, 1], pca_results[:, 2])
        ax.set_title('PCA')
        ax.set_xlabel('1st eigenvector')
        ax.set_ylabel('2nd eigenvector')
        ax.set_zlabel('3rd eigenvector')
        plt.show()

    if args.kmeans == 0:
        print('No Sampling was requested')
        exit()

    elif args.kmeans >= len(original_list):
        print('Not enough points to downsample')
        exit()

    else:
        kmeans = KMeans(n_clusters=int(args.kmeans), random_state=0).fit(pca_results)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # plot the kmeans results with the original points
        if args.dimension == '2':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(pca_results[:, 0], pca_results[:, 1], c=labels)
            ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5, c='r')
            ax.set_title('KMeans')
            ax.set_xlabel('1st eigenvector')
            ax.set_ylabel('2nd eigenvector')
            plt.show()
        elif args.dimension == '3':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pca_results[:, 0], pca_results[:, 1], pca_results[:, 2], c=labels)
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=150, linewidths=5, c='r')
            ax.set_title('KMeans')
            ax.set_xlabel('1st eigenvector')
            ax.set_ylabel('2nd eigenvector')
            ax.set_zlabel('3rd eigenvector')
            plt.show()





