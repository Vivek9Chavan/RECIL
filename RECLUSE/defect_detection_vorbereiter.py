import os
import shutil
import re
import json
import numpy as np
import torch


mvtec_path = '/mnt/logicNAS/Exchange/vivek/Defect_Clustering_dino/MVTec/'


#for each folder in the mvtec_path
for folder in os.listdir(mvtec_path):
    print(folder)
    for def_type in os.listdir(os.path.join(mvtec_path, folder)):
        print(def_type)
        if def_type.endswith('.png'):
            continue
        else:
            for file in os.listdir(os.path.join(mvtec_path, folder, def_type)):
                if file.endswith('.png'):
                    file_name = file.split('.')[0]
                    # add the defect type to the file name
                    new_file_name = file_name + '_' + def_type + '.png'
                    # rename the file
                    os.rename(os.path.join(mvtec_path, folder, def_type, file), os.path.join(mvtec_path, folder, def_type, new_file_name))
                    #move the file to the main folder
                    shutil.move(os.path.join(mvtec_path, folder, def_type, new_file_name), os.path.join(mvtec_path, folder, new_file_name))
            #remove the empty folder
            os.rmdir(os.path.join(mvtec_path, folder, def_type))





