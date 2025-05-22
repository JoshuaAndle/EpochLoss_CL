"""
Does standard subnetwork training on all tasks

"""

from __future__ import division, print_function

import torch
import torch.nn as nn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
import warnings
import copy
import time
from itertools import islice
from torch.optim.lr_scheduler  import MultiStepLR
from math import floor
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import torch.utils.data as D



from torchvision import datasets, transforms
from PIL import Image


# To prevent PIL warnings.
warnings.filterwarnings("ignore")

###General flags
FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--generator', type=str, default="", help='AI Generator to process')
FLAGS.add_argument('--folder_path', type=str, default=".", help='Path to AI generator directories.')
FLAGS.add_argument('--output_folder', type=str, default="processedSingleGenerators", help='Id of current run.')
FLAGS.add_argument('--task_num', type=int, default=0, help="Which task to process")


def main():
    args = FLAGS.parse_args()
   
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('-'*100, flush=True)   



    
    reverseDict = {}
    natureClassDict = torch.load((args.folder_path + "/natureClassDict.pt"))
    for key, value in natureClassDict.items():
        reverseDict[value[0]] = key


    # for generator in ["ADM", "BigGAN", "Midjourney", "glide", "stable_diffusion_v_1_5"]:
    for expType in ['single_generator']:
        # task = args.task_num
        for split in ['train', 'val']:
            for imgType in ['ai', 'nature']:
                images_as_tensor(args.generator, expType, split, imgType, reverseDict, args)



# def images_as_tensor(generator, expType, split, imgType, reverseDict, args, target_size=(64, 64)):
def images_as_tensor(generator, expType, split, imgType, reverseDict, args, target_size=64):

    transformAI = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  
      ])
    ### Need to resize shorter edge of rectangular images to 64, then crop the longer edge to match
    transformNature = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  
      ])

    directory = os.path.join(".", generator, expType, str(args.task_num), split, imgType)
    saveDirectory = os.path.join(".", args.output_folder, generator, expType, str(args.task_num), split, imgType)
    image_tensors, labels = [], []
    print("Load Directory: ", directory, flush=True)
    print("Save Directory: ", saveDirectory, flush=True)

    # Parse all images in the directory
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            ### Convert label from name string to integer value
            labelString = filename.split("_")[0]
            label = int(reverseDict[labelString])-1

            img_path = os.path.join(directory, filename)
            # ### Since I processed as pngs to loss, need to remove alpha channel
            # img = Image.open(img_path).convert('RGB')  
            # if imgType == "ai":
            #     img_tensor = transformAI(img)
            # elif imgType == "nature":
            #     img_tensor = transformNature(img)
            # else:
            #     print("Wrong imgType for transform")


            # image_tensors.append(img_tensor)
            # labels.append(label)


            try:
                img = Image.open(img_path)
                img.info.pop("icc_profile", None)  # Remove ICC profile if present
                img = img.convert('RGB')  

                if imgType == "ai":
                    img_tensor = transformAI(img)
                elif imgType == "nature":
                    img_tensor = transformNature(img)
                else:
                    print(f"Wrong imgType for transform: {imgType}")
                    continue 

                image_tensors.append(img_tensor)
                labels.append(label)
            
            ### Added because a few images throw decompression errors
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

    if len(image_tensors) == 0:
        print("No valid images found. Skipping tensor creation.")
        return None, None



    # Stack tensors and encode labels
    image_tensors = torch.stack(image_tensors)
    label_tensor = torch.LongTensor(np.array(labels,dtype=int)).view(-1)
    # if split == "train":
    print("Size: ", image_tensors.shape)
    # image_tensors = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])(image_tensors)


    if not os.path.exists(saveDirectory):
        os.makedirs(saveDirectory, exist_ok=True)
    torch.save(image_tensors, os.path.join(saveDirectory, 'X.pt'))
    torch.save(label_tensor, os.path.join(saveDirectory, 'y.pt'))

    return image_tensors, labels
    
    
if __name__ == '__main__':
    main()

