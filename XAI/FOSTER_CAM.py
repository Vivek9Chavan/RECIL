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

from utils.inc_net import FOSTERNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM

test_image = '/mnt/Welle/messy_10.jpg'

checkpoint='/mnt/checkpoints/GrIL/FOSTER_InVar_Lite224_messy_12.pth'

model = FOSTERNet('resnet18', False)
increments = [10, 10, 10, 10, 10, 4, 7, 2, 14, 5, 3, 9, 6]

model.update_fc(10)
model.update_fc(20)
model.update_fc(30)
model.update_fc(40)
model.update_fc(50)
model.update_fc(54)
model.update_fc(61)
model.update_fc(63)
model.update_fc(77)
model.update_fc(82)
model.update_fc(85)
model.update_fc(94)
model.update_fc(100)


# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
#print(model)

net = model.convnets[-1] # resnet18(pretrained=True)
finalconv_name = 'layer4'
net._modules.get(finalconv_name).register_forward_hook(hook_feature)

print(net)


model.eval()
model.cpu()

classes = os.listdir('/mnt/InVar_100_lite/train_resized_224/')
#print(classes)

import os
import torch
import torchvision.transforms as transforms
from PIL import Image


# Define the transformation to apply to the input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

# load test image
img_pil = Image.open(test_image)
img_tensor = preprocess(img_pil) #preprocess the image
img_variable = Variable(img_tensor.unsqueeze(0))

# Run the forward pass with no gradient computation
with torch.no_grad():
    logit = model(img_variable)['logits']
    print(logit)
    h_x = F.softmax(logit, dim=1)

# Get the predicted class
prob, predicted_class = torch.max(h_x, 1)

def get_activations(model, image, layer_name='layer4'):
    model.eval()
    with torch.no_grad():
        x = image.unsqueeze(0)
        # Get the activations of the final convolutional layer
        activations = None
        for name, module in model._modules.items():
            x = module(x)
            if name == layer_name:
                activations = x.detach().clone()
        return activations

#print(get_activations(model._modules['convnet'], img_tensor, 'layer4'))
#print(get_activations(model._modules['convnet'], img_tensor, 'layer4').shape)


print("Predicted class:", predicted_class.item())

class_dict = {'Adapter': 0, 'Behaelterrot': 1, 'Biegeteil': 2, 'Blechteildick': 3,
              'Blechteilgeschweisst': 4, 'BlechteilmitBohrungen': 5, 'BlechteilmitschlechtenBohrungen': 6,
              'Brandschutzschalter': 7, 'Brille': 8, 'BronzeUnterlegscheibe': 9, 'Dichtung1': 10,
              'Dichtung10weiss': 11, 'Dichtung11schwarz': 12, 'Dichtung12schwarz': 13,
              'Dichtung13tuerkis': 14, 'Dichtung14schwarz': 15, 'Dichtung2': 16, 'Dichtung3': 17,
              'Dichtung4': 18, 'Dichtung5gruen': 19, 'Dichtung6schwarz': 20, 'Dichtung7grau': 21,
              'Dichtung8rot': 22, 'Dichtung9schwarz': 23, 'Dichtungsring': 24, 'Dieselfilter': 25,
              'Duese': 26, 'Duesedichtend': 27, 'Dueseplatte1': 28, 'Dueseplatte2': 29, 'Eckverbinder': 30,
              'EckverbinderZinklegierung': 31, 'Elektronikgehaeuse': 32, 'ElektrostreifenStecker': 33,
              'EthernetModule': 34, 'Fuehrung': 35, 'Fuehrungsschraube': 36, 'Gelenk': 37,
              'GeruestSteinmitFeder': 38, 'Geruestklemme': 39, 'Gewindestange': 40,
              'Gleitlagergehaeuse': 41, 'Holzteil': 42, 'HuelseGewinde': 43, 'HuelsemitSchlitz': 44,
              'Hutmutter': 45, 'Kabelbinder': 46, 'KabelbindermitKlappe': 47, 'Kappe': 48,
              'Klemmbuchse': 49, 'Kochtopfdeckelgriff': 50, 'Kunststoffgehaeuse': 51,
              'Kunststoffschraubenfeder': 52, 'Kurbel': 53, 'Laserzuschnitt': 54, 'Lineal': 55,
              'Messrohr': 56, 'Messschieber': 57, 'Metallschraubenfeder': 58, 'Nutsteinprofil': 59,
              'Oelschraube': 60, 'Pumpenrohr': 61, 'Rohrgitter': 62, 'Rundblech': 63, 'Schlauch': 64,
              'Schluessel': 65, 'Schraube1': 66, 'Schraube2': 67, 'Schraube3': 68, 'Schraube4': 69,
              'Schraube5': 70, 'Schraube6': 71, 'Schraube7': 72, 'SchraubemitOelbohrungen': 73,
              'Senkschraube1': 74, 'Staender': 75, 'Stahldeckel': 76, 'Stahlschnur': 77,
              'Steckdosegruen': 78, 'Steckergruen1': 79, 'Steckgruen2': 80, 'Steckteil': 81, 'Stoepsel': 82,
              'TerminalBlockgrau': 83, 'TerminalBlockgrau2': 84, 'TerminalBlockgrau3': 85,
              'TerminalBlockgruen': 86, 'TerminalBlockrot': 87, 'USBKabel': 88, 'Unterlegscheibe1': 89,
              'Unterlegscheibe2': 90, 'Verschlussschraube': 91, 'Welle': 92, 'Winkel': 93,
              'Zahnradrohr': 94, 'Zange': 95, 'gebogeneStange': 96, 'lackiertesKunststoffteil': 97,
              'variableHuelse': 98, 'variableHuelse2': 99}

class_order = [20, 57, 34, 79, 68, 14, 66, 39, 13, 86, 72, 65, 37, 52, 6, 46, 89, 95, 0, 80, 30, 93, 9, 59,
                       21, 27, 32, 77, 29, 31, 53, 84, 88, 2, 70, 62, 36, 61, 23, 54, 82, 11, 8, 50, 91, 47, 15, 19,
                       56, 51, 17, 87, 24, 4, 67, 12, 38, 90, 73, 18, 55, 3, 78, 44, 10, 96, 64, 58, 35, 75, 97, 94,
                       49, 48, 28, 25, 5, 81, 74, 45, 42, 60, 1, 83, 22, 98, 26, 33, 40, 41, 99, 43, 85, 92, 76, 71,
                       69, 63, 7, 16]

class_dict = {v: k for k, v in class_dict.items()}
class_dict = {k: class_dict[k] for k in class_order}

#take classes 21-30 from the dict
class_na = {k: class_dict[k] for k in list(class_dict)}

#take only the class names

class_names = list(class_na.values())

print(class_names)

#print top 5 predictions
probs, idxs = h_x.topk(5)
probs = probs.squeeze()
idxs = idxs.squeeze()
for i in range(5):
    #print class names, ids and probabilities
    print('{:.3f} -> {} ({})'.format(probs[i], class_names[idxs[i]], idxs[i]))


idx = idxs[0].item()
print('idx: ', idx)


params = list(model.parameters())
#for i in range(len(params)):
#    print(i, params[i].size())


weight_softmax = np.squeeze(params[-4].data.numpy())

print('weight_softmax.shape', weight_softmax.shape)
print('features_blobs[0].shape', features_blobs[0].shape)
#weights = model._modules['convnet'].state_dict()['layer4'].cpu().numpy()
#print('weights.shape', weights.shape)


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        #print('modified idx: ', idx)
        #print('Weight softmax idx shape: ', weight_softmax[idx].shape)
        #print(weight_softmax[idx])
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


for i in range(100):
    idm = [i]
    CAMs = returnCAM(features_blobs[0], weight_softmax, idm)

    img = cv2.imread(test_image)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)

    result = heatmap * 0.4 + img * 0.5
    cv2.imwrite('/mnt/checkpoints/GrIL/foster_cam/' + str(i) + '.jpg', result)
