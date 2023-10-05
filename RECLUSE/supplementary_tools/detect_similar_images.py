"""
The current configuration is to identify the similar images from a SINGLE CLASS.
To identify the similar images from all classes, loop through the classes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import json
from sklearn.decomposition import PCA
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description='Detect similar images using PCA results')
    parser.add_argument('--feats_path', type=str, default='/mnt/logicNAS/Exchange/ImageNet_100/', help='Path to the stored features')
    parser.add_argument('--max_tolerance', type=float, default=0.1, help='Maximum tolerance for the similarity')
    parser.add_argument('--output_dir', type=str, help='path to the output directory')
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

if __name__ == '__main__':
    args = parse_arguments()

    similar_images_list = []

    for folder in os.listdir(args.feats_path):
        print('folder: ', folder)
        fold = os.path.join(args.feats_path, folder)

        #get original list of features
        preds = get_original_list(fold)
        pred_array = np.concatenate(preds[:,1], axis=0)
        names = preds[:,0]

        #pca downsample
        pca = PCA(n_components=3, svd_solver='full', random_state=404543)
        pca_results = pca.fit_transform(pred_array)

        #find entries within max_tolerance
        tolerance = args.max_tolerance

        #find similar images
        similar_images = []
        for i in range(len(pca_results)):
            for j in range(i+1, len(pca_results)):
                if np.linalg.norm(pca_results[i] - pca_results[j]) < tolerance:
                    similar_images.append((names[i], names[j]))

        print(">>>Number of similar images: ", len(similar_images))
        print(similar_images)

        if len(similar_images) > 0:

            #append to list
            similar_images_list.append(folder)
            similar_images_list.append(similar_images)

    #save to json
    with open(os.path.join(args.output_dir, 'similar_images.json'), 'w') as f:
        json.dump(similar_images_list, f)

