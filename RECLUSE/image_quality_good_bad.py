# This script is used to sort the images into good and bad categories
# The images are sorted by the user, by clicking on the 'good' or 'bad' button
# The images are then moved to the corresponding directory

import os
import shutil
import random
import numpy as np

random.seed(404543)

working_dir = '/home/chavvive/eiba_good_bad_sort/sort/ground_truth_1/'

working_dir_good = '/home/chavvive/eiba_good_bad_sort/images_good_GT1'
working_dir_bad = '/home/chavvive/eiba_good_bad_sort/images_bad_GT1'

# if does not exist, create the directory
if not os.path.exists(working_dir_good):
    os.makedirs(working_dir_good)
if not os.path.exists(working_dir_bad):
    os.makedirs(working_dir_bad)

# create a gui to display the images one by one
import matplotlib.pyplot as plt
import tkinter as tk
import PIL.Image, PIL.ImageTk


# when good button is clicked, add '_good' to the file name
def good_button_clicked(filename):
    filename_new = filename.replace('.jpg', '_good.jpg')
    os.rename(os.path.join(working_dir, filename), os.path.join(working_dir, filename_new))
    # move the file to the good directory
    shutil.move(os.path.join(working_dir, filename_new), os.path.join(working_dir_good, filename_new))
    gui.destroy()


# when bad button is clicked, add '_bad' to the file name
def bad_button_clicked(filename):
    filename_new = filename.replace('.jpg', '_bad.jpg')
    os.rename(os.path.join(working_dir, filename), os.path.join(working_dir, filename_new))
    # move the file to the bad directory
    shutil.move(os.path.join(working_dir, filename_new), os.path.join(working_dir_bad, filename_new))
    gui.destroy()

def previous_button_clicked(filename_prev):
    #get input from user to rename the file
    # if the file contains '_bad' in its name, replace it with '_good'
    if '_bad' in filename_prev:
        filename_new = filename_prev.replace('_bad', '_good')
        os.rename(os.path.join(working_dir, filename_prev), os.path.join(working_dir, filename_new))
    # if the file contains '_good' in its name, replace it with '_bad'
    elif '_good' in filename_prev:
        filename_new = filename_prev.replace('_good', '_bad')
        os.rename(os.path.join(working_dir, filename_prev), os.path.join(working_dir, filename_new))
    gui.destroy()


if __name__ == '__main__':

    files = os.listdir(working_dir)

    # loop through each file in working_dir
    for i in files:

        filename = os.path.join(working_dir, i)
        index = files.index(i)
        gui = tk.Tk()
        # add two buttons to the gui: good and bad
        good_button = tk.Button(gui, text='good', command=lambda: good_button_clicked(i))
        # green background
        good_button.config(bg='green')
        good_button.pack()
        bad_button = tk.Button(gui, text='bad', command=lambda: bad_button_clicked(i))
        # red background
        bad_button.config(bg='red')
        bad_button.pack()

        '''
        # add a button to open previous image
        previous_button = tk.Button(gui, text='Correct Previous', command=lambda: previous_button_clicked(files[index-1]))
        previous_button.pack()
        # located at the left of the gui
        previous_button.place(x=0, y=0)
        '''

        # load the first image in the gui
        img = PIL.Image.open(filename)
        img = img.resize((1200, 800))
        imgtk = PIL.ImageTk.PhotoImage(img)
        label = tk.Label(gui, image=imgtk)
        label.pack()
        # title the gui with the file name
        gui.title(i)
        #open the gui, after a button is clicked, close the gui
        gui.mainloop()
        print(index)




