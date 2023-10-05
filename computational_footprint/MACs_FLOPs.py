"""
Calculating the computational complexity of the given approach using the ptflops library
The results obtained by this method are verified and cross-referenced with those in published literature

More info: https://github.com/sovrasov/flops-counter.pytorch

On average, 1MACs = 2*Flops
"""

import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from torchvision import models
import sys
import traceback
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchsummary import summary


with torch.no_grad():
    try:
        with torch.cuda.device(0):
            net = models.resnet18()

            net = net.to(device=0)

            summary(net, (3, 224, 224))

            net2 = fasterrcnn_resnet50_fpn()

            macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=False,
                                                     print_per_layer_stat=False, verbose=False)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    except AssertionError:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]

        print('An error occurred on line {} in statement {}'.format(line, text))
        exit(1)