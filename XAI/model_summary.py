import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import random
import pandas as pd
from PIL import Image
import cv2
import torchsummary
from utils.inc_net import DERNet, IncrementalNet

print(DERNet)

from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM

model = DERNet('resnet18', False)
model.update_fc(10)
model.update_fc(20)
#model.update_fc(10)

checkpoint = '/mnt/checkpoints/GrIL/der_net_InVar100_lite224_white_1.pth'

model.load_state_dict(torch.load(checkpoint))

print(model)

#print(model._modules['convnets']._modules['0']._modules['layer4'])

net = model._modules['convnets']._modules['0']
