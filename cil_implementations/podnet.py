import math
import logging
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from cil_implementations.base import BaseLearner
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import tensor2numpy
from torchvision import models, transforms
import argparse


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
import ptflops
from ptflops import get_model_complexity_info


import random
import gc
from time import time

epochs = 160  # 160 epochs
lrate = 0.1

ft_epochs = 20  # 20 epochs
ft_lrate = 0.005 # 0.005
batch_size = 16 #16

lambda_c_base = 5
lambda_f_base = 1
nb_proxy = 10
weight_decay = 5e-4
num_workers = 8



# create a wandb run
import wandb

wandb.init(project="GrIL_Paper",
           name = 'PODNet_6_months_Invar_Lite_224')

#define gradcam
def Gradcam(model, img, target_layer, device):
    model.eval()
    img = img.to(device)
    target_layer = target_layer
    grad_cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)
    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    target_category = None
    grayscale_cam = grad_cam(input_tensor=img, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]
    cam = show_cam_on_image(img.cpu(), grayscale_cam)
    return cam


class PODNet(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = CosineIncrementalNet(args['convnet_type'], pretrained=False, nb_proxy=nb_proxy)
        self._class_means = None
        self._increment = args['increments']

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1

        increments = self._increment
        self._total_classes = self._known_classes + increments[self._cur_task]
        print('Total classes: ', self._total_classes)

        self.task_size = self._total_classes - self._known_classes
        self._network.update_fc(self._total_classes, self._cur_task)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                              mode='train', appendent=self._get_memory())
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self._train(data_manager, self.train_loader, self.test_loader)
        print('====> Samples per class: ', self.samples_per_class)
        self.build_rehearsal_memory(data_manager, self.samples_per_class) #VC: Commented out to save memory

        macs, params = get_model_complexity_info(self._network.to(self._device), as_strings=True,
                                                 print_per_layer_stat=False, verbose=True)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        wandb.log({"Macs": macs, "Params": params})
        
        print('Macs: ', macs)
        print('Params: ', params)

    def _train(self, data_manager, train_loader, test_loader):
        if self._cur_task == 0:
            self.factor = 0
        else:
            self.factor = math.sqrt(self._total_classes / (self._total_classes - self._known_classes))
        logging.info('Adaptive factor: {}'.format(self.factor))
        wandb.log({'factor': self.factor})

        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            network_params = self._network.parameters()
        else:
            ignored_params = list(map(id, self._network.fc.fc1.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, self._network.parameters())
            network_params = [{'params': base_params, 'lr': lrate, 'weight_decay': weight_decay},
                              {'params': self._network.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
        self._run(train_loader, test_loader, optimizer, scheduler, epochs)

        if self._cur_task == 0:
            return
        logging.info('Finetune the network (classifier part) with the undersampled dataset!')
        if self._fixed_memory:
            finetune_samples_per_class = self._memory_per_class

            
            self._construct_exemplar_unified(data_manager, finetune_samples_per_class)
            #self.get_exemplar_dino()
            #self.get_online_set()

        else:
            #if self._memory_size // self._known_classes <= 41:
            finetune_samples_per_class = self._memory_size // self._known_classes
            #else:
                #finetune_samples_per_class = 41
            self._reduce_exemplar(data_manager, finetune_samples_per_class)
            self._construct_exemplar(data_manager, finetune_samples_per_class)

        finetune_train_dataset = data_manager.get_dataset([], source='train', mode='train',
                                                          appendent=self._get_memory())
        finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)
        logging.info('The size of finetune dataset: {}'.format(len(finetune_train_dataset)))

        ignored_params = list(map(id, self._network.fc.fc1.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self._network.parameters())
        network_params = [{'params': base_params, 'lr': ft_lrate, 'weight_decay': weight_decay},
                          {'params': self._network.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
        optimizer = optim.SGD(network_params, lr=ft_lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=ft_epochs)
        self._run(finetune_train_loader, test_loader, optimizer, scheduler, ft_epochs)

        if self._fixed_memory:
            self._data_memory = self._data_memory[:-self._memory_per_class * self.task_size]
            self._targets_memory = self._targets_memory[:-self._memory_per_class * self.task_size]
            assert len(np.setdiff1d(self._targets_memory, np.arange(0, self._known_classes))) == 0, 'Exemplar error!'

    def _run(self, train_loader, test_loader, optimizer, scheduler, epk):
        for epoch in range(1, epk + 1):
            self._network.train()
            lsc_losses = 0.
            spatial_losses = 0.
            flat_losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                outputs = self._network(inputs)

                logits = outputs['logits']
                features = outputs['features']
                fmaps = outputs['fmaps']
                lsc_loss = nca(logits, targets)

                spatial_loss = 0.
                flat_loss = 0.
                if self._old_network is not None:
                    with torch.no_grad():
                        old_outputs = self._old_network(inputs)
                    old_features = old_outputs['features']
                    old_fmaps = old_outputs['fmaps']
                    flat_loss = F.cosine_embedding_loss(features, old_features.detach(),
                                                        torch.ones(inputs.shape[0]).to(
                                                            self._device)) * self.factor * lambda_f_base
                    spatial_loss = pod_spatial_loss(fmaps, old_fmaps) * self.factor * lambda_c_base

                loss = lsc_loss + flat_loss + spatial_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                wandb.log({'loss': loss})

                lsc_losses += lsc_loss.item()
                spatial_losses += spatial_loss.item() if self._cur_task != 0 else spatial_loss
                flat_losses += flat_loss.item() if self._cur_task != 0 else flat_loss

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                wandb.log({'lsc_loss': lsc_losses, 'spatial_loss': spatial_losses, 'flat_loss': flat_losses})

                total += len(targets)

            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info1 = 'Task {}, Epoch {}/{} (LR {:.5f}) => '.format(
                self._cur_task, epoch, epk, optimizer.param_groups[0]['lr'])
            info2 = 'LSC_loss {:.2f}, Spatial_loss {:.2f}, Flat_loss {:.2f}, Train_acc {:.2f}, Test_acc {:.2f}'.format(
                lsc_losses / (i + 1), spatial_losses / (i + 1), flat_losses / (i + 1), train_acc, test_acc)
            logging.info(info1 + info2)
            wandb.log({'train_acc': train_acc, 'test_acc': test_acc})

            
            ######
            #íf test acc is higher than 99% break the loop
            if test_acc >= 99:
                print('Early Stopping: test_acc > 99%')
                break
                

        # empty the cache
        #gc.collect()
        #torch.cuda.empty_cache()
        print('Memory allocated: ', torch.cuda.memory_reserved())
        print('Cached memory: ', torch.cuda.memory_cached())




        from torch.autograd import Variable

        dummy_input = Variable(torch.randn(1, 3, 224, 224)).cuda()

        
        # save the model
        #torch.save(self._network.state_dict(), './checkpoints/Dataset_Sampling/pod_net_Invar_lite224_{}.pth'.format(self._cur_task))

        # save the architecture as pt file
        torch.save(self._network, './checkpoints/GrIL/pod_net_Dino_{}.pth'.format(self._cur_task))

        #network = '/mnt/1TBNVME/Green_Incremental_Learning/checkpoints/pod_net_12.pt'
        #model = torch.jit.load(network)

        # export onnx model
        #torch.onnx.export(self._network, dummy_input, './checkpoints/pod_net_{}.onnx'.format(self._cur_task))

        #save the updated model
        #torch.jit.save(self._network, './checkpoints/pod_net_{}.pt'.format(self._cur_task))

        
        



def pod_spatial_loss(old_fmaps, fmaps, normalize=True):
    '''
    a, b: list of [bs, c, w, h]
    '''
    loss = torch.tensor(0.).to(fmaps[0].device)
    for i, (a, b) in enumerate(zip(old_fmaps, fmaps)):
        assert a.shape == b.shape, 'Shape error'

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        a_h = a.sum(dim=3).view(a.shape[0], -1)  # [bs, c*w]
        b_h = b.sum(dim=3).view(b.shape[0], -1)  # [bs, c*w]
        a_w = a.sum(dim=2).view(a.shape[0], -1)  # [bs, c*h]
        b_w = b.sum(dim=2).view(b.shape[0], -1)  # [bs, c*h]

        a = torch.cat([a_h, a_w], dim=-1)
        b = torch.cat([b_h, b_w], dim=-1)

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss
        wandb.log({'pod_spatial_loss': loss / len(fmaps)})

    return loss / len(fmaps)


def nca(
        similarities,
        targets,
        class_weights=None,
        focal_gamma=None,
        scale=1.0,
        margin=0.6,
        exclude_pos_denominator=True,
        hinge_proxynca=False,
        memory_flags=None,
):
    margins = torch.zeros_like(similarities)
    margins[torch.arange(margins.shape[0]), targets] = margin
    similarities = scale * (similarities - margin)

    if exclude_pos_denominator:
        similarities = similarities - similarities.max(1)[0].view(-1, 1)

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
                    targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.)

        loss = torch.mean(losses)
        return loss

    return F.cross_entropy(similarities, targets, weight=class_weights, reduction="mean")