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

from PCA_and_visualise import get_original_list

import random
random.seed(404543)
torch.manual_seed(404543)

# image extensions, input the application extension in args.image_extension
image_extension = ['.jpg', '.jpeg', 'JPEG', '.png', '.bmp', '.gif', '.tif', '.tiff']

def select_img(selected_pth, ext):
    selected_img = []

    for line in selected_pth:
        line = line.strip()
        if line.find('.pth') != -1:
            line = line[:line.find('.pth')] + '.' + ext
            selected_img.append(line)
    return selected_img

# argparse
def arg_parse():
    parser = argparse.ArgumentParser(description='Sample Dataset')
    parser.add_argument('--feats_path', type=str, default='/mnt/DINO/', help='Path where the features are saved')
    parser.add_argument('--images_path', type=str, default='/mnt/rain/', help='Path where the images are saved')
    parser.add_argument('--image_extension', type=str, default='JPEG', help='Image extension')
    parser.add_argument('--save_dir', type=str, default='/mnt/train_20_DINO/', help='Path where to save the selected images')
    parser.add_argument('--pca_dim', type=int, default=32, help='Dimension of PCA')
    parser.add_argument('--num_images_per_class', type=int, default=20, help='number of images per class')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    print(args)

    folders = os.listdir(args.images_path)

    for folder in folders:
        # create a folder for each class in the save_dir
        if not os.path.exists(os.path.join(args.save_dir, folder)):
            os.makedirs(os.path.join(args.save_dir, folder))

        folder_img = os.path.join(args.images_path, folder)
        folder_features = os.path.join(args.feats_path, folder)
        folder_target = os.path.join(args.save_dir, folder)

        print('=== folder: ', folder, '===')

        preds = get_original_list(folder_features)
        pred_array = np.concatenate(preds[:,1], axis=0)

        names = preds[:,0]


        #pca = PCA(n_components=args.pca_dim, svd_solver='full', random_state=404543)
        #pca_results = pca.fit_transform(pred_array)

        kmeans = KMeans(n_clusters=args.num_images_per_class, random_state=404543, n_init=100)
        #clusters = kmeans.fit(pca_results)
        clusters = kmeans.fit(pred_array)

        # get the cluster centers
        cluster_centers = kmeans.cluster_centers_

        #distances = cdist(cluster_centers, pca_results, 'euclidean')
        distances = cdist(cluster_centers, pred_array, 'euclidean')

        closest = []

        for i in range(len(cluster_centers)):
            closest_point = np.argmin(distances[i])

            # if the closest point already exists in closest, select the next closest point
            if names[closest_point] in closest:
                print('closest point already exists: ', names[closest_point])
                # find the next closest point
                next_closest_point = np.argmin(distances[i][distances[i] > distances[i][closest_point]])
                print('Next Closest point: ', names[next_closest_point])
                closest_point = next_closest_point


            closest.append(names[closest_point])

        selected_img = select_img(closest, args.image_extension)

        print(selected_img)


        for file in os.listdir(folder_img):
            if file in selected_img:
                print('copying: ', file)
                shutil.copy(os.path.join(folder_img, file), folder_target)
            else:
                print('skipping: ', file)






