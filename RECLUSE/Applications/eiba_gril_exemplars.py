import shutil
import re
import os
import sys
import argparse
import numpy as np

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from torchvision import transforms
from torchvision.models import resnet50

from utlilties import utils, vision_transformer as vits
from PIL import Image

import sys
import pandas as pd

import ptflops

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from PCA_and_visualise import get_original_list

import random
random.seed(404543)
torch.manual_seed(404543)


transform_1 = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def get_vectors(image, model, transform):
    """
    Get the feature vectors for all images in the path.
    """
    img = Image.open(image).convert('RGB')
    # get the length and width of the image
    #width, height = img.size
    #print(width, height)
    #aspect_ratio = width/height
    #transform = transforms.Compose([transforms.Resize((224, int(224*aspect_ratio))),
                                    #transforms.ToTensor(),
                                    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    img = transform(img)
    # add batch dimension
    img = img.unsqueeze(0)
    # Get the feature vector for the image
    #GPU:
    vector = model(img.cuda(non_blocking=True))
    #CPU:
    #vector = model(img)
    #making rank 0
    vector = nn.functional.normalize(vector, dim=1, p=2)
    #get image name
    target = image[image.rfind('/')+1:]
    return vector, target

def get_original_list(folder):
    original_list = []
    for filename in list(os.listdir(folder)):
        vect, targ = get_vectors(os.path.join(folder, filename), model, transform_1)
        if vect is not None:
            original = vect
            original = original.detach().cpu().numpy()
            original = [targ, original]
            original_list.append(original)
    return np.array(original_list, dtype=object)

def select_img(selected_pth, ext):
    selected_img = []

    for line in selected_pth:
        line = line.strip()
        if line.find('.pth') != -1:
            line = line[:line.find('.pth')] + '.' + ext
            selected_img.append(line)
    return selected_img

parent_dir = '/mnt/1TBNVME/vivek/EIBA_GrIL_exp/'

month = ['2021_11', '2021_12', '2022_01', '2022_02', '2022_03', '2022_04', '2022_05', '2022_06', '2022_07', '2022_08', '2022_09', '2022_11']


model = vits.vit_small()
model.cuda()
utils.load_pretrained_weights(model=model, pretrained_weights=True, checkpoint_key='teacher', model_name='vit_small', patch_size=16)
model.eval()

for mon in month:
    increment = os.path.join(parent_dir, mon)
    print(increment)
    cumulative_dir = os.path.join(increment, 'cumulative')
    exemplar_dir = os.path.join(increment, 'exemplars')
    for cls in os.listdir(cumulative_dir):
        print(cls)
        if not os.path.exists(os.path.join(exemplar_dir, cls)):
            os.makedirs(os.path.join(exemplar_dir, cls))
        cls_dir = os.path.join(cumulative_dir, cls)
        cls_images = os.listdir(cls_dir)
        preds = get_original_list(cls_dir)

        pred_array = np.concatenate(preds[:, 1], axis=0)

        names = preds[:, 0]
        #print('Names: ', names)

        pca = PCA(n_components=32, svd_solver='full', random_state=404543)
        pca_results = pca.fit_transform(pred_array)

        print(pca_results.shape)

        kmeans = KMeans(n_clusters=20, random_state=404543, n_init=100)
        clusters = kmeans.fit(pca_results)

        # get the cluster centers
        cluster_centers = kmeans.cluster_centers_
        print('cluster centers: ', len(cluster_centers))

        distances = cdist(cluster_centers, pca_results, 'euclidean')

        print('Distances: ', distances.shape)

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

        print('closest: ', len(closest))
        selected_img = closest #select_img(closest, 'jpg')

        print('selected images: ', selected_img)
        print('length of selected images: ', len(selected_img))

        for img in selected_img:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(exemplar_dir, cls, img))





