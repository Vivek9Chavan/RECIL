"""
Aim: To save feature vectors for the images from the original dataset

"""
import sys
import argparse

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

from utlilties import utils, vision_transformer as vits

import os
import shutil
import torch
import torch.nn as nn


def get_folders(path):
    folders = []
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            folders.append(folder)
    return folders

import re

def get_file_num(path):
    file_num = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if re.search(r'\.(jpg|jpeg|png|bmp|tiff)$', file):
                file_num += 1
    return file_num


def get_file_name(path, i):
    """
    get file name in path
    :param path: path
    :param i: index
    :return: filename
    """
    file_list = os.listdir(path)
    file_list.sort()
    return os.path.splitext(file_list[i])[0]


# extract features from a pretrained model
# the model is loaded from a checkpoint
# the features are extracted from a dataset
# the features are saved in a file

def extract_feature_pipeline(args, title, model):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = ReturnIndexDataset(os.path.join(args.feats_path), transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Data loaded with {len(dataset_train)} train imgs.")

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)

    if utils.get_rank() == 0:
        print("Saving features...")
        train_features = nn.functional.normalize(train_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    # save features and labels
    if args.feats_path and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.feats_path, title+ ".pth"))
        #torch.save(train_labels.cpu(), os.path.join(args.feats_path, "trainlabels.pth"))

    return train_features, train_labels



@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    # metric_logger is a class that helps to print the progress of the training
    metric_logger = utils.MetricLogger(delimiter="  ")
    # features is a tensor that will contain the features of the dataset
    features = None

    # idx is the index of the class
    idx = 0
    i = 0
    # for each batch of the dataset
    for samples, index in metric_logger.log_every(data_loader, 10):
        #Move to GPU
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # if multiscale is True, we use the multiscale function
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone() #features: To be used!

        """
        print(dump_features_path)

        print(feats.shape)
        print("Saving this feature...")
        torch.save(feats.cpu(), os.path.join(dump_features_path, name + ".pth"))

        i+= 1
        if i == num_files:
            idx+= 1
            i = 1
        """

        # downsize the features to (1,3)
        feats = feats.view(feats.size(0), -1)


        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")


        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )

        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


# This is a class that inherits from the ImageFolder class in the torchvision.datasets module.
# It is used to return the index of the image in the dataset.
# This is used in the code to return the index of the image in the dataset.

class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int, help='Per-GPU batch-size') # we use 1 currently, to ensure that the image name and the feature name are identical and that there are no issues with feature correspondance

    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
                        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--original_images_path', default='/home/chavvive/ImageNet100/train/', type=str)
    parser.add_argument('--feats_path', default='/home/chavvive/ImageNet100/feats/', type=str)
    parser.add_argument('--img_extension', default=False, type=bool)
    args = parser.parse_args()

    path_original_images = args.original_images_path # path to original dataset with images sorted by class
    path_for_features = args.feats_path # path to folder where you want to copy files

    # create path2 if it doesn't exist
    if not os.path.exists(path_for_features):
        os.makedirs(path_for_features)

    # for industrial_100 dataset
    image_type = ['industrial', 'messy', 'white', 'with_hand']

    #wandb.config.update(args)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # Currently loaded features are NOT being used for Sampling
    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))

        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))


    #load model:

    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)

    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # To extract features

    #take first 20 folders alphabetically from the original_images_path
    folder_list = os.listdir(path_original_images)
    folder_list.sort()
    folder_list = folder_list[:20]

    print('Folder list: ', folder_list)
    """
    for folder in os.listdir(path_original_images): # loop through folders
    """
    for folder in folder_list:
        if args.img_extension == False:
            os.mkdir(path_for_features + '/' + folder) # create folder with same name in path2
            for file in os.listdir(path_original_images + '/' + folder): # loop through files in folder
                if file.endswith('.JPEG') or file.endswith('.jpg') or file.endswith('.png') or file.endswith('.JPG'):
                    # if file is jpg or png, then n = 3
                    if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.JPG'):
                        n = 4
                    elif file.endswith('.JPEG'):
                        n = 5
                    shutil.copy(path_original_images + '/' + folder + '/' + file, path_for_features + '/' + folder)
                    title = folder + '_' +file[:-n] # print file name without extension
                    """
                    for i in os.listdir(args.dump_features):
                        if os.path.isdir(path2 + "/" + i):
                            name3 = i
                    title = name3 + "_" + title
                    """
                    print("title: ", title)
                    train_features, train_labels = extract_feature_pipeline(args, title, model)
                    os.remove(path_for_features + '/' + folder + '/' + file) # remove file from folder
                    #time.sleep(2) # pause for 2 seconds
            os.rmdir(path_for_features + '/' + folder) # delete folder after all files have been copied


        # Use this if the dataset has intra-class subsets, like the industrial_100 dataset
        if args.img_extension == True:
            for type in image_type:
                try:
                    os.mkdir(path_for_features + '/' + folder) # create folder with same name in path2
                except:
                    continue
                os.mkdir(path_for_features + '/' + folder + '/' + type)
                for file in os.listdir(path_original_images + '/' + folder + '/' +  type): # loop through files in folder
                    if file.endswith('.JPEG') or file.endswith('.jpg') or file.endswith('.png'):
                        # if file is jpg or png, then n = 3
                        if file.endswith('.jpg') or file.endswith('.png'):
                            n = 4
                        elif file.endswith('.JPEG'):
                            n = 5
                        shutil.copy(path_original_images + '/' + folder + '/' + type + '/' + file, path_for_features + '/' + folder + '/' + type) # copy file to path2
                        title = folder + '_' + file[:-n] # print file name without extension
                        """
                        for i in os.listdir(args.dump_features):
                            if os.path.isdir(path2 + "/" + i):
                                name3 = i
                        title = name3 + "_" + title
                        """
                        print("title: ", title)
                        train_features, train_labels = extract_feature_pipeline(args, title, model)
                        os.remove(path_for_features + '/' + folder + '/' + type + '/' +  file) # remove file from folder
                        #time.sleep(2) # pause for 2 seconds
                os.rmdir(path_for_features + '/' + folder + '/' + type) # delete folder after all files have been copied
                os.rmdir(path_for_features + '/' + folder)  #

    dist.barrier()

    #create all the folders in the feats folder
    for folder in os.listdir(path_original_images):
        if not os.path.exists(path_for_features + '/' + folder):
            os.makedirs(path_for_features + '/' + folder)

        if args.img_extension == True:
            for type in image_type:
                if not os.path.exists(path_for_features + '/' + folder + '/' + type):
                    os.makedirs(path_for_features + '/' + folder + '/' + type)

    # match the folder names with the labels before _ in the file name
    for folder in os.listdir(path_for_features):
        fold = folder + '_'
        for file in os.listdir(path_for_features):
            if file.endswith('.pth') and file.startswith(fold):
                #move the file to the folder
                shutil.move(path_for_features + '/' + file, path_for_features + '/' + folder)
                os.rename(path_for_features + '/' + folder + '/' + file, path_for_features + '/' + folder + '/' + file[len(fold):])
                #print("renamed: ", file)

    print("============ Done ============")
