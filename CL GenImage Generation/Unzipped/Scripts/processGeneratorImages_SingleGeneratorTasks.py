"""
Does standard subnetwork training on all tasks

"""

from __future__ import division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
import warnings
import copy
import time
import torch.nn as nn
import numpy as np
import torch
from itertools import islice
from torch.optim.lr_scheduler  import MultiStepLR
from math import floor
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import torch.utils.data as D

from PIL import Image

# To prevent PIL warnings.
warnings.filterwarnings("ignore")

###General flags
FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--generator', type=str, default="", help='AI Generator to process')
FLAGS.add_argument('--folder_path', type=str, default=".", help='Path to AI generator directories.')
FLAGS.add_argument('--output_folder', type=str, default=".", help='Path to save subset of dataset to')
FLAGS.add_argument('--task_num', type=int, default=0, help="Which task to process")

# FLAGS.add_argument('--removal_by_task', nargs='+', type=str, default=['None'], help= 'Dictates which removal metric to use for each task in the sequence')


#***# Save structure: Runid is the experiment, different task orders are subdirs that share up to the last common task so that the task can be reused/located just by giving the runid and task sequence and can be shared between multiple alternative orders for efficiency
###    Basically this just means 6 nested directories, which are nested in order of task order for the given experiment. So all subdirs of the outermost directory 2 have task 2 as the first task and can share the final dict from task 2 amongst eachother for consistency and efficiency
def main():
    args = FLAGS.parse_args()
   
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('-'*100, flush=True)   



    aiClassDict, natureClassDict, tinyImagenetClasses = {}, {}, []
    ### Generated in produceSubsetDicts.py which needs to be run first
    aiClassDict = torch.load((args.folder_path + "/aiClassDict.pt"))
    natureClassDict = torch.load((args.folder_path + "/natureClassDict.pt"))






    num_classes_by_task = 100


    process_images(num_classes_by_task, aiClassDict, natureClassDict, args)





def convertImageName(imgName, classDict, generator=None, imgSplit='train', imgType='ai'):
    if imgType == "ai":
        classIndex, imageIndex = -1, -1
        if generator in ['ADM', 'BigGAN', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5']:
            classIndex = 0
            imageIndex = 2
        elif generator in ['glide', 'VQDM']:
            classIndex = 4
            imageIndex = 6

        splitName = imgName.split('_')
        classNumber = str(int(splitName[classIndex]))
        classString = classDict[classNumber][0]
        
        ### Get rid of the file format text and remove any padding
        imageNumber = splitName[imageIndex].split('.')[0]
        convertedImageNumber = str(int(imageNumber))

        newName = (classString + "_" + generator + "_" + convertedImageNumber + ".png")
        return classString, newName

    elif imgType == "nature":
        if imgSplit == "train":
            splitName = imgName.split('_')
            classString = splitName[0]
            imageNumber = splitName[1].split('.')[0]
            newName = (classString + "_" + imageNumber + ".png")
            return classString, newName

        elif imgSplit == "val":
            splitName = imgName.split('_')
            imageNumber = int(splitName[2].split('.')[0])
            unpaddedImageNumber = str(imageNumber)

            ### The image numbers start at 1, so offset the index by 1 for indexing into list
            classNumber = classDict['valLabels'][imageNumber-1]
            classString = classDict[classNumber][0]
            newName = (classString + "_" + unpaddedImageNumber + ".png")
            return classString, newName





def process_images(num_classes, aiClassDict, natureClassDict, args, target_size=(64, 64)):
    """
    Processes images from the input folder, resizing and saving a subset to the output folder.

    Parameters:
        input_folder (str): Path to the folder containing the images.
        output_folder (str): Path to save the resized images.
        selected_classes (list): List of class IDs to include.
        target_size (tuple): The target size for the images (width, height).
    """


    classDict = torch.load("./generatorClassesFull.pt")
    joint_classes = classDict['joint']

    classConversionDict = {}

    # generatorNames = ["ADM", "BigGAN", "Midjourney", "glide", "stable_diffusion_v_1_5"]
    # for generator in generatorNames:

    generator = args.generator


    ### We access the disjoint class dicts of each generator in the list to pull the appropriate classes for each task
    classGeneratorNames = ["ADM", "BigGAN", "Midjourney", "glide", "stable_diffusion_v_1_4", "VQDM"]

    # for taskID, classGen in enumerate(classGeneratorNames):
    ### We take the list of classes to use based on the task number
    generator_classes = classDict[classGeneratorNames[args.task_num]]

    print("Getting classes for taskID ", args.task_num, " corresponding to disjoint classes of generator ", classGeneratorNames[args.task_num], flush=True)
    for imageset in ['train', 'val']:
        # for imagetype in ['nature']:
        for imagetype in ['ai', 'nature']:
            if imagetype == "ai":
                classConversionDict = aiClassDict
            elif imagetype == "nature":
                classConversionDict = natureClassDict

            input_folder = os.path.join(args.folder_path, generator, "raw", imageset, imagetype)
            task_folder = os.path.join(args.folder_path, generator, "single_generator", str(args.task_num), imageset, imagetype)

            print("Processing for input folder: ", input_folder, flush=True)
            if not os.path.exists(task_folder):
                os.makedirs(task_folder, exist_ok=True)

            for filename in os.listdir(input_folder):
                # Extract class ID from filename
                # class_id = filename.split('_')[0]
                classString, newFileName = convertImageName(filename, classConversionDict, generator, imageset, imagetype) 

                ### Check if this class is in the selected subset
                if classString in generator_classes:
                    input_path = os.path.join(input_folder, filename)
                    output_path = os.path.join(task_folder, newFileName)

                    try:
                        # Open, resize, and save the image
                        with Image.open(input_path) as img:
                            if imagetype == "ai":
                                img = img.resize(target_size)
                            img.save(output_path, format='PNG')
                    except Exception as e:
                        print(f"Error processing {input_path}: {e}")




    
    
if __name__ == '__main__':
    
    main()

