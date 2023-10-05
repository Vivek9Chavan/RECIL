"""
The current configuration is to identify the outliers from a SINGLE CLASS.
To remove the outliers from all classes, loop through the classes and remove the outliers from each class.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
from sklearn.decomposition import PCA
import os
import torch


def parse_arguments():
    parser = argparse.ArgumentParser(description='Remove outliers from PCA results')
    parser.add_argument('--feats_path', type=str, default='/mnt/ImageNet_100/', help='Path to the stored features')
    parser.add_argument('--max_points_to_remove', type=int, default=3, help='Maximum number of points to remove')
    parser.add_argument('--output_dir', type=str, default='.', help='path to the output directory')
    return parser.parse_args()

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


def remove_outliers(pca_results, names, points_to_remove):
    # remove the outliers
    # calculate the mean of the data
    mean = np.mean(pca_results, axis=0)

    # calculate the distance of each point to the mean
    distances = np.sqrt(np.sum((pca_results - mean)**2, axis=1))

    # if the distance is greater than 6 standard deviations, remove the point
    std = np.std(distances)
    mask = distances < 6*std
    pca_results_mod = pca_results[mask]
    names_mod = names[mask]
    # if more points are removed
    if len(pca_results_mod) >= len(pca_results)-points_to_remove:
        return pca_results_mod, names_mod, mask
    else:
        print('removed too many points')
        return pca_results, names, mask


if __name__ == '__main__':
    args = parse_arguments()

    outliers_list = []

    #loop through all the folders in the feats_path
    for folder in os.listdir(args.feats_path):
        print('folder: ', folder)
        fold = os.path.join(args.feats_path, folder)
        # get original list of features
        preds = get_original_list(fold)
        pred_array = np.concatenate(preds[:,1], axis=0)
        names = preds[:,0]

        #pca downsample
        pca = PCA(n_components=3, svd_solver='full', random_state=404543)
        pca_results = pca.fit_transform(pred_array)

        # remove outliers
        pca_results_mod, names_mod, mask = remove_outliers(pca_results, names, args.max_points_to_remove)
        print('Identified {} points'.format(len(pca_results_mod)-len(mask)))

        #if no outliers are removed
        if len(pca_results_mod) == len(pred_array):
            print('No clear outliers found. Change the criteria or study the data distribution further')

        # print the names of the points that were removed
        print(names[~mask])

        # if the number of points removed is greater than 0, save the results
        if len(names[~mask]) > 0:

            # append names[~mask] to the outliers_list
            outliers_list.append(folder)
            outliers_list.append(names[~mask])

            """
            Plotting outliers is commented out for looping through all classes
            """

            """
            #perform 3D PCA
            #pca = PCA(n_components=3, svd_solver='full', random_state=404543)
            #pca_results_3d = pca.fit_transform(pca_results)
    
            pca_results_3d = pca_results_mod
    
            #plot the points
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pca_results_3d[:,0], pca_results_3d[:,1], pca_results_3d[:,2], c='b', marker='o')
    
            #show removed points
            ax.scatter(pca_results_3d[~mask,0], pca_results_3d[~mask,1], pca_results_3d[~mask,2], c='r', marker='o', s=200)
    
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            plt.show()
            """
    # save the outliers_list
    with open(os.path.join(args.output_dir, 'outliers_list.json'), 'w') as f:
        json.dump(outliers_list, f)
    print('Done')






