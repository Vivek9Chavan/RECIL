"""
This is meant to create a shuffled train / val / test split of the image dataset from the original dataset.
"""

import os
import json
import copy
import numpy as np
import shutil
import pandas

def train_test_split():
    print("########### Train Test Val Script started ###########")

    target_dir = '/mnt/4TB_HDD/chavvive/vivek/ILSVRC/Data/CLS-LOC/'

    original_dir = '/mnt/4TB_HDD/chavvive/vivek/ILSVRC/Data/CLS-LOC/train/'

    classes_dir = list(os.listdir(os.path.join(original_dir)))

    # change as needed:
    val_ratio = 0.10
    test_ratio = 0.10

    for cls in classes_dir:
        # Creating partitions of the data after shuffling
        print("$$$$$$$ Class Name " + cls + " $$$$$$$")
        src = original_dir + cls #'/white' # Folder to copy images from

        allFileNames = os.listdir(src)
        np.random.shuffle(allFileNames)
        train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                                  [int(len(allFileNames) * (1 - (val_ratio + test_ratio))),
                                                                   int(len(allFileNames) * (1 - test_ratio)),
                                                                   ])

        train_FileNames = [src + '/' + name for name in train_FileNames.tolist()]
        val_FileNames = [src + '/' + name for name in val_FileNames.tolist()]
        test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]

        print('Total images: '+ str(len(allFileNames)))
        print('Training: '+ str(len(train_FileNames)))
        print('Validation: '+  str(len(val_FileNames)))
        print('Testing: '+ str(len(test_FileNames)))

        try:
            # # Creating Train / Val / Test folders (One time use)
            os.makedirs(target_dir + '/train/' + cls)
            os.makedirs(target_dir + '/val/' + cls)
            os.makedirs(target_dir + '/test/' + cls)
        except:
            next

        # Copy-pasting images
        for name in train_FileNames:
            shutil.copy(name, target_dir + '/train/' + cls)

        for name in val_FileNames:
            shutil.copy(name, target_dir + '/val/' + cls)

        for name in test_FileNames:
            shutil.copy(name, target_dir + '/test/' + cls)

    print("########### Train Test Val Script Ended ###########")


if __name__ == '__main__':
    train_test_split()

