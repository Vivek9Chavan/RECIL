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



model = vits.vit_small()
#model = resnet50(pretrained=True)
model.cuda()

ptflops.get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
#utils.load_pretrained_weights(model=model, pretrained_weights=True, checkpoint_key='teacher', model_name='vit_base', patch_size=16)
model.eval()

transform_1 = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

#transform_2 will have 90 degree clockwise rotation
transform_2 = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.RandomRotation(90),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

#transform_3 will have 90 degree counter clockwise rotation
transform_3 = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.RandomRotation(270),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

#transform_4 will have 180 degree rotation
transform_4 = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.RandomRotation(180),
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
    #vector = model(img.cuda(non_blocking=True))
    #making rank 0
    vector = nn.functional.normalize(vector, dim=1, p=2)
    #get image name
    target = image
    #print(target)
    #print(vector)S
    return vector, target

if __name__ == '__main__':

    image_path = '/mnt/ImageNet_100_2023/train/'

    save_path = '/mnt/ImageNet_100_2023/Pruning_feats/DINO/'


    for folders in os.listdir(image_path):
        print(folders)
        if not os.path.exists(save_path + folders):
            os.makedirs(save_path + folders)
        for i in os.listdir(os.path.join(image_path, folders)):

            #image = df.iloc[i,0]

            image = image_path + folders + '/' + i
            print(image)
            for i in range(2,3):
                tr = 'transform_' + str(i)
                vector, target = get_vectors(image, model, transform=transform_1)

                target = target.split('/')[-1]
                # remove the .jpg extension
                target = target.split('.')[0]
                target = target + '_' + str(i)
                print(target)

                if not os.path.exists(save_path + folders + '/' + target + '.pth'):
                    #save the vector as a pth file
                    torch.save(vector, save_path + folders + '/' + target + '.pth')





