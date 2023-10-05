# This code is meant to establish the shelly plug baseline and then measure the power consumption of the system during training.

'''
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from time import sleep
import wandb
import requests

#requests.get("http://192.168.33.1/meter/0").json()['power']

# get the power consumption of the system
def get_power():
    return requests.get("http://192.168.33.1/meter/0").json()['power']

total_power = []

def establish_baseline(time = 3600):
    # establish the baseline
    # get the power consumption of the system
    for i in range(time):
        print('Power: ', get_power())
        total_power.append(get_power())
        sleep(0.5)
    # get average power consumption
    average_power = np.mean(total_power)
    #print(average_power)
    print("Baseline established: " + str(average_power))
    return average_power

if __name__ == '__main__':
    establish_baseline()
    #print(total_power)

'''
########################################################################################################################

# This code is meant to log the power consumption of the system during training.

import time
import datetime
import requests
import logging
import os
import pandas as pd


def log_power(name):
    root = '/mnt/1TBNVME/MVIP_GrIL/logs/'
    logging.basicConfig(filename=os.path.join(root, name), level=logging.INFO)
    t = time.time()
    while True:
        try:
            if time.time() > t + 1:
                t = time.time()
                power = dict(requests.get("http://192.168.33.1/meter/0").json())['power']
                logging.info(time.ctime(t) + ' | ' + str(t) + ' | ' + str(power))

                df = pd.read_csv(os.path.join(root, name), sep='|', header=None)
                df.to_csv(os.path.join(root, name.split('.')[0] + '.csv'), index=False)
        except:
            pass
        time.sleep(0.25)

if __name__ == '__main__':
    root = '/mnt/1TBNVME/MVIP_GrIL/logs/'
    name = 'power_log_24_01.log'

    # save as a csv file

    if not os.path.exists(root):
        os.makedirs(root)

    log_power(os.path.join(root, name))






