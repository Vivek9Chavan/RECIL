import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from cil_implementations.base import BaseLearner
from utils.inc_net import DERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
import random
import gc
from time import time
import ptflops



# set max_split_size to a large number to avoid memory error
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=8000





# ensure deterministic behaviour
np.random.seed(404543)
random.seed(404543) # set seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.manual_seed(404543)
torch.cuda.manual_seed_all(404543)

EPSILON = 1e-8

init_epoch= 200  #200
init_lr=0.1
init_milestones=[60,120,170]
init_lr_decay=0.1
init_weight_decay=0.0005


epochs = 170   #170
lrate = 0.1
milestones = [80, 120,150]
lrate_decay = 0.1
batch_size = 16
weight_decay=2e-4
num_workers=8
T=2

# create a wandb run
import wandb
import ptflops
from ptflops import get_model_complexity_info



wandb.init(project="GrIL_Paper",
           name = 'DER_StanfordCars_SUP')


print('DER Net')

class DER(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = DERNet(args['convnet_type'], pretrained=True)
        
        # print the flops and params of the network


    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1

        increments = args['increments']
        self._total_classes = self._known_classes + increments[self._cur_task]
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task>0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))


    
                               
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)





        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
            
            
    def train(self):
        self._network.train()
        self._network.convnets[-1].train()
        #VC: removing self._network.module.convnets...
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network.convnets[i].eval()
                #VC: self._network.module.convnets[i].eval()


    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        print('current task: ', self._cur_task)
        if self._cur_task == 0:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), momentum=0.9,
                                  lr=init_lr, weight_decay=init_weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones,
                                                       gamma=init_lr_decay)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), lr=lrate, momentum=0.9, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes-self._known_classes)
            else:
                self._network.weight_align(self._total_classes-self._known_classes)

            
    def _init_train(self,train_loader,test_loader, optimizer, scheduler):
        #print the classes
        print('Known classes: {}'.format(self._known_classes))
        #print the number of classes
        print('Total number of classes: {}'.format(self._total_classes))
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']

                loss=F.cross_entropy(logits,targets) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)

            if epoch % 5 == 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, init_epoch, losses / len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, init_epoch, losses / len(train_loader), train_acc, test_acc)
                
                wandb.log({'Task': self._cur_task, 'Epoch': epoch + 1, 'Loss': losses / len(train_loader),
                           'Train_acc': train_acc, 'Test_acc': test_acc})

            prog_bar.set_description(info)

            


            # empty the cache
            gc.collect()
            torch.cuda.empty_cache()

            
        # empty the cache
        gc.collect()
        torch.cuda.empty_cache()
        print('Memory allocated: ',torch.cuda.memory_reserved())
        print('Cached memory: ',torch.cuda.memory_cached())
        
        input_tensor = torch.randn(3, 224, 224, dtype=torch.float32, device=self._device)

        macs, params = get_model_complexity_info(self._network.to(self._device), as_strings=True,
                                                 print_per_layer_stat=False, verbose=True)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        wandb.log({"Macs": macs, "Params": params})

        logging.info('Computational complexity: {}'.format(macs))
        logging.info('Number of parameters: {}'.format(params))

        # save the model
        torch.save(self._network.state_dict(), f='./checkpoints/GrIL/der_Imagenet_{}.pth'.format(self._cur_task))
        """
        task_0_classes = ['n01632777', 'n01798484', 'n01729322', 'n01955084', 'n01855672']

        from PIL import Image
        import torchvision.transforms as transforms
        import os

        tr_zahnradrohr = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        for cla in task_0_classes:
            img_dir = '/mnt/1TBNVME/ImageNet_100_2023/train/' + cla
            save_pth_dir = '/mnt/1TBNVME/ImageNet_100_2023/saved_feats_imgnet/' + '/' + str(
                self._cur_task) + '/' + cla
            if not os.path.exists(save_pth_dir):
                os.makedirs(save_pth_dir)
            for img_name in os.listdir(img_dir):
                img_pth = os.path.join(img_dir, img_name)
                img = Image.open(img_pth)
                img = tr_zahnradrohr(img)
                img = img.unsqueeze(0)
                img = img.to(self._device)
                feat = self._network(img)['features']
                feat = feat.cpu().detach().numpy()
                # save as pth
                torch.save(feat, save_pth_dir + '/' + img_name[:-4] + '.pth')

        print('Done Saved them Features!')
        """
        # save the model in onnx format
        #torch.onnx.export(self._network, inputs,  f='./checkpoints/der_net_{}.onnx'.format(self._cur_task))
        #wandb.save('der_net_{}.onnx'.format(self._cur_task))

        logging.info(info)
        # empty the cache
        gc.collect()
        torch.cuda.empty_cache()


    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        # print the classes
        print('Training on classes: {}'.format(self._known_classes))
        # print the number of classes
        print('Number of classes: {}'.format(self._total_classes))
        
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.
            losses_clf=0.
            losses_aux=0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs= self._network(inputs)
                logits,aux_logits=outputs["logits"],outputs["aux_logits"]
                targets = targets.long()  # VC: Added this line
                loss_clf=F.cross_entropy(logits,targets)
                
                aux_targets = targets.clone()
                aux_targets=torch.where(aux_targets-self._known_classes+1>0,aux_targets-self._known_classes+1,0)
                loss_aux=F.cross_entropy(aux_logits,aux_targets)
                loss=loss_clf+loss_aux

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux+=loss_aux.item()
                losses_clf+=loss_clf.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch%5==0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader),losses_clf/len(train_loader),losses_aux/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader),losses_clf/len(train_loader),losses_aux/len(train_loader), train_acc)
                

                wandb.log({'Task': self._cur_task, 'Epoch': epoch + 1, 'Loss': losses / len(train_loader),
                           'Loss_clf': losses_clf / len(train_loader), 'Loss_aux': losses_aux / len(train_loader),
                           'Train_acc': train_acc, 'Test_acc': test_acc})
                
            # log losses and accuracies using wandb

            prog_bar.set_description(info)
            # empty the cache
            gc.collect()
            torch.cuda.empty_cache()

        input_tensor = torch.randn(3, 224, 224, dtype=torch.float32, device=self._device)

        macs, params = get_model_complexity_info(self._network.to(self._device), as_strings=True,
                                                 print_per_layer_stat=False, verbose=True)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        wandb.log({"Macs": macs, "Params": params})

        logging.info('Computational complexity: {}'.format(macs))
        logging.info('Number of parameters: {}'.format(params))

        # save the model
        #torch.save(self._network.state_dict(), f='./checkpoints/GrIL/der_Imagenet_{}.pth'.format(self._cur_task))

        
        
        logging.info(info)

        # empty the cache
        gc.collect()
        torch.cuda.empty_cache()

        # clear gpu allocated memory
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        torch.cuda.reset_peak_memory_stats()
                
        print('Memory allocated: ', torch.cuda.memory_allocated())
        print('Cached memory: ', torch.cuda.memory_cached())
        

     