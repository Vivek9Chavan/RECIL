# Code to evaluate the EIBA data
# Given the folder, the code will extract the features and create a plot
# The plot will be opened in a new browser window

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
from RECLUSE.utlilties import utils, vision_transformer as vits
import wandb
from PIL import Image
from sklearn.decomposition import PCA
import pandas as pd

import random
#random.seed(0)
#set torch seed for reproducibility
#torch.manual_seed(0)
from matplotlib import pyplot as plt
import plotly.express as px # We use plotly instead of matplotlib to create the plot

###########################################################################################################

def get_vectors(image, model, device):
    """
    Get the feature vectors for all images in the path.
    """
    img = Image.open(image).convert('RGB')
    # get the length and width of the image
    #width, height = img.size
    #print(width, height)
    #aspect_ratio = width/height
    #transform = pth_transforms.Compose([pth_transforms.Resize((224, int(224*aspect_ratio))), pth_transforms.ToTensor(), pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    img = transform(img)
    # add batch dimension
    img = img.unsqueeze(0)
    # Get the feature vector for the image
    if device == 'cuda':
        vector = model(img.cuda(non_blocking=True))
    if device == 'cpu':
        vector = model(img)
    #making rank 0
    vector = nn.functional.normalize(vector, dim=1, p=2)
    #get image name
    target = image
    print(target)
    #print(vector)S
    return vector, target

def get_all_vectors(image_path, model):
    """
    from image path, extract vectors, downsample and plot a plotly scatter plot
    get all images in the path
    """
    all_vectors = []
    for i in os.listdir(image_path):
        image = image_path + i
        #print(image)
        vector, target = get_vectors(image, model, device)
        vector = vector.detach().cpu().numpy()
        # cut the target to only have the name of the image
        target = target.split('/')[-1]
        #remove the extension
        target = target.split('.')[0]
        #store all in all_vectors
        vectors= [target, vector]
        all_vectors.append(vectors)
    return np.array(all_vectors, dtype=object)


if __name__ == '__main__':

    ############################################################################################################
    # Change parameters here:
    ############################################################################################################

    parser = argparse.ArgumentParser('Extract features, downsample and visualise results')
    parser.add_argument('--original_images_path', default='/home/chavvive/TEC_Allianz/Manually_Sampled/', type=str) # Path to the original images
    parser.add_argument('--dimension', type=str, default='2', help='2D or 3D') # For data visualisation
    parser.add_argument('--device', type=str, default='cpu', help='cuda or CPU') # GPU or CPU
    args = parser.parse_args()

    ############################################################################################################

    print(args)

    # Model setup:
    model = vits.vit_small()
    device = args.device
    if device == 'cuda':
        model.cuda()
    if device == 'cpu':
        model = model.cpu()
    utils.load_pretrained_weights(model=model, pretrained_weights=True, checkpoint_key='teacher', model_name='vit_small', patch_size=16)
    model.eval()

    transform = pth_transforms.Compose([
       pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    original_list = get_all_vectors(args.original_images_path, model)
    #print(original_list)
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

    # plot the PCA results using plotly
    if args.dimension == '2':
        # plotly scatter plot
        df = pd.DataFrame(pca_results, columns=['x', 'y'])
        df['image'] = original_list[:,0]
        fig = px.scatter(df, x='x', y='y', hover_data=['image'], color='image')
        fig.update_traces(marker=dict(size=15, line=dict(width=1, color='DarkSlateGrey')))
        fig.show()
    elif args.dimension == '3':
        #plotly scatter plot
        df = pd.DataFrame(pca_results, columns=['x', 'y', 'z'])
        df['image'] = original_list[:,0]
        fig = px.scatter_3d(df, x='x', y='y', z='z', hover_data=['image'], color='image')
        #add border to the points
        fig.update_traces(marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')))
        fig.show()









