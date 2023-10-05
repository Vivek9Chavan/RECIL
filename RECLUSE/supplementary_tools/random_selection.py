"""
This is meant to select a random subset of image files from a folder.
The images can be used as a comparison against the subset selected using DINO.
"""


import os
import json
import random
import shutil


def get_random_files(path, num_files, class_name):
    files = os.listdir(path)
    random.shuffle(files)
    return [(file, class_name) for file in files[:num_files]]


def get_random_files_from_folders(path, num_files):
    folders = os.listdir(path)
    random.shuffle(folders)
    files = []
    for folder in folders:
        files.extend(get_random_files(os.path.join(path, folder), num_files, folder))
    return files


def get_file_details(path, files):
    file_details = []
    for file, class_name in files:
        file_details.append({
            'name': file,
            'path': os.path.join(path, file),
            'class_name': class_name
        })
    return file_details


def write_json(file_details, file_name):
    with open(file_name, 'w') as f:
        json.dump(file_details, f)


if __name__ == '__main__':
    path = '/mnt/train/Adapter/'
    target_path = '/mnt/train/'
    num_files = 20
    files = get_random_files_from_folders(path, num_files)
    file_details = get_file_details(path, files)
    write_json(file_details, './random_selection.json')
    # Copy the images to the target path
    for file_detail in file_details:
        shutil.copy(file_detail['path'], target_path + file_detail['class_name'])
    print('Done')


