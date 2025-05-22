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
FLAGS.add_argument('--folder_path', type=str, default=".", help='Path to AI generator directories.')
FLAGS.add_argument('--output_folder', type=str, default=".", help='Path to save subset of dataset to')

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
    


    ### Set up or get the dictionaries used to convert class IDs from Imagenet format
    ### Note: the ai and nature images use different conversion dicts
    if not os.path.isfile((args.folder_path + "/aiClassDict.pt")):
        import json

        ### Load JSON file and save as pytorch file for consistency
        with open((args.folder_path + '/imagenet_class_index.json'), 'r') as file:
            aiClassDict = json.load(file)

        torch.save(aiClassDict, (args.folder_path + "/aiClassDict.pt"))
    else:
        aiClassDict = torch.load((args.folder_path + "/aiClassDict.pt"))


    if not os.path.isfile((args.folder_path + "/natureClassDict.pt")):
        # Open the file in read mode
        with open((args.folder_path + '/map_clsloc.txt'), 'r') as file:
            # Read each line in the file
            for line in file:
                # Print each line
                parts = line.strip().split()
                if len(parts) == 3:  
                    key = parts[1]  
                    value = [parts[0], parts[2]] 
                    natureClassDict[key] = value

        valLabels = []
        # Open the file in read mode
        with open((args.folder_path + '/ILSVRC2015_clsloc_validation_ground_truth.txt'), 'r') as file:
            # Read each line in the file
            for line in file:
                # Print each line
                valLabels.append(line.strip())
        natureClassDict['valLabels'] = valLabels
        torch.save(natureClassDict, (args.folder_path + "/natureClassDict.pt"))
    else:
        natureClassDict = torch.load((args.folder_path + "/natureClassDict.pt"))



    ### Get the classes used from Tiny Imagenet, since we use this subset for pretraining the network so need to omit them
    if not os.path.isfile((args.folder_path + "/tinyImagenetClasses.pt")):
        # Open the file in read mode
        with open((args.folder_path + '/tinyImagenetWnids.txt'), 'r') as file:
            # Read each line in the file
            for line in file:
                # Print each line
                tinyImagenetClasses.append(line.strip())
        torch.save(tinyImagenetClasses, (args.folder_path + "/tinyImagenetClasses.pt"))
    else:
        tinyImagenetClasses = torch.load((args.folder_path + "/tinyImagenetClasses.pt"))

    print("Length of tiny imagenet classes: ", len(tinyImagenetClasses), " with values:\n ", tinyImagenetClasses, flush=True)







    num_classes_by_task = 100
    produceDicts(num_classes_by_task, aiClassDict, natureClassDict, tinyImagenetClasses, args)
    # updateDicts(num_classes_by_task, aiClassDict, natureClassDict, tinyImagenetClasses, args)
    



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



### Get the raw list of classes used in each subset for each generator
def getGeneratorClasses(generatorNames, aiClassDict, natureClassDict, args):
    generatorClassDict = {}
    for generator in generatorNames:
        for imageset in ['train', 'val']:
            for imagetype in ['ai', 'nature']:
                classList, classDict = [], {}

                inputfolder = os.path.join(args.folder_path, generator, 'raw', imageset, imagetype)


                if imagetype == "ai":
                    classDict = aiClassDict
                else:
                    classDict = natureClassDict

                for filename in os.listdir(inputfolder):
                    # Extract class ID from filename
                    classString, _ = convertImageName(filename, classDict, generator, imageset, imagetype) 

                    if classString not in classList:
                        classList.append(classString)

                keyName = (generator + "_" + imageset + "_" + imagetype)
                generatorClassDict[keyName] = classList

    return generatorClassDict










### Extends an existing dictionary of generator classes to add new generators
def updateDicts(num_classes, aiClassDict, natureClassDict, tinyImagenetClasses, args):
    classList, classDict = [], {}

    generatorNames = ["ADM", "BigGAN", "Midjourney", "glide", "stable_diffusion_v_1_4", "VQDM"]
    generatorClassDict = getGeneratorClasses(generatorNames, aiClassDict, natureClassDict, args)


    partialGeneratorClassesDict = torch.load("./generatorClasses.pt")
    ### We are removing this generator since there were issues with the zip files provided by the dataset
    del(partialGeneratorClassesDict['stable_diffusion_v_1_5'])

    print("Keys in partial classes after removing SD1.5: ", partialGeneratorClassesDict.keys())
    finalGeneratorClassesDict = {}
    dictValues = list(generatorClassDict.values())

    finalGeneratorClassesDict['joint'] = partialGeneratorClassesDict['joint']



    ### For each generator, get a viable list of classes that was not used yet and is present in all of its subsets
    for generator in generatorNames:
        generatorCandidateClasses = []
        ### If the dataset was already included in the partial set of generators, then just copy its values
        if generator in partialGeneratorClassesDict.keys():
            print("Classes already determined for generator: ", generator, flush=True)
            finalGeneratorClassesDict[generator] = partialGeneratorClassesDict[generator]
        ### Otherwise get a subset of classes not yet used in the other generators' subsets or in TinyImagenet
        else:
            print("Classes not found for generator: ", generator, flush=True)
            for imageset in ['train', 'val']:
                for imagetype in ['ai', 'nature']:
                    keyName = (generator + "_" + imageset + "_" + imagetype)
                    ### Add any classes not already present in the current generator list
                    generatorCandidateClasses.extend(list(set(generatorClassDict[keyName]) - set(generatorCandidateClasses)))

            ### Remove as candidates all classes that have already been assigned for use
            for classList in list(finalGeneratorClassesDict.values()):
                generatorCandidateClasses = list(set(generatorCandidateClasses) - set(classList))

            ### Remove as candidates any classes pretrained on in Tiny Imagenet
            generatorCandidateClasses = list(set(generatorCandidateClasses) - set(tinyImagenetClasses))

            print("Number of remaining candidates for ", generator, ": ", len(generatorCandidateClasses), flush=True)
            finalGeneratorClassesDict[generator] = generatorCandidateClasses[:num_classes]

    torch.save(finalGeneratorClassesDict, "./generatorClassesFull.pt")

    for key in finalGeneratorClassesDict:
        print("\n\n\n Number of classes for ", key, ": ", len(finalGeneratorClassesDict[key]))
        print(finalGeneratorClassesDict[key])



### Produce a dictionary of classes to use for each generator, drawn from the full set of Imagenet classes

def produceDicts(num_classes, aiClassDict, natureClassDict, tinyImagenetClasses, args):
    classList, classDict = [], {}

    generatorNames = ["ADM", "BigGAN", "Midjourney", "glide", "stable_diffusion_v_1_5"]
    generatorClassDict = getGeneratorClasses(generatorNames, aiClassDict, natureClassDict, args)




    finalGeneratorClassesDict = {}
    dictValues = list(generatorClassDict.values())

    ### Get the classes that are present in all subsets of the data to be used in determining a valid joint subset
    jointClasses = dictValues[0]
    for i in range(1,len(dictValues)):
        jointClasses = list(set(jointClasses) & set(dictValues[i]))
    ### For joint class setup we want to make sure the classes were previously included in the pretraining task Tiny Imagenet
    jointClasses = list(set(jointClasses) & set(tinyImagenetClasses))

    print("Number of valid joint classes shared by all sets: ", len(jointClasses), flush=True)

    finalGeneratorClassesDict['joint'] = jointClasses[:num_classes]



    ### For each generator, get a viable list of classes that was not used yet and is present in all of its subsets
    for generator in generatorNames:
        generatorCandidateClasses = []
        for imageset in ['train', 'val']:
            for imagetype in ['ai', 'nature']:
                keyName = (generator + "_" + imageset + "_" + imagetype)
                ### Add any classes not already present in the current generator list
                generatorCandidateClasses.extend(list(set(generatorClassDict[keyName]) - set(generatorCandidateClasses)))

        ### Remove as candidates all classes that have already been assigned for use
        for classList in list(finalGeneratorClassesDict.values()):
            generatorCandidateClasses = list(set(generatorCandidateClasses) - set(classList))

        ### Remove as candidates any classes pretrained on in Tiny Imagenet
        generatorCandidateClasses = list(set(generatorCandidateClasses) - set(tinyImagenetClasses))

        print("Number of remaining candidates for ", generator, ": ", len(generatorCandidateClasses), flush=True)
        finalGeneratorClassesDict[generator] = generatorCandidateClasses[:num_classes]

    torch.save(finalGeneratorClassesDict, "./generatorClasses.pt")

    for key in finalGeneratorClassesDict:
        print("\n\n\n Number of classes for ", key, ": ", len(finalGeneratorClassesDict[key]))
        print(finalGeneratorClassesDict[key])


    # print("Number of classes found: ", len(classList), flush=True)


    # classIdx = 0

    # ### Get a disjoint set of classes for each generator for the disjoint set
    # ### First we get a joint set to be used in an alternative setup for all generators
    # for generator in generatorNames:
    #     generatorClasses = classList[classIdx:classIdx+num_classes]
    #     generatorClassDict[generator] = generatorClasses
    #     classIdx += num_classes
    #     print("Storing ", len(generatorClassDict[generator]), " classes for generator: ", generator, flush=True)
    # torch.save(generatorClassDict, "./generatorClasses.pt")





    
    
if __name__ == '__main__':
    
    main()

