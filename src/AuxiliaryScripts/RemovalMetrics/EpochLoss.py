import os
import copy
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from AuxiliaryScripts import corruptions 

"""
Implements the EpochLoss metric for removal of training data based on initial per-sample performance
"""
class EpochLoss_Method():
    
    def __init__(self, args:argparse.Namespace, model:nn.Module, extra_loader:data.DataLoader, sampledict:dict):
        self.args = args
        self.model = model
        self.extraloader= extra_loader
        self.sampledict = sampledict



    ### Produce a mask of samples to be removed based on metric settings
    def gen_data_mask(self):
        labels = []
        IDs = []
        for data, target, ID in self.extraloader:
            labels.extend(target.numpy())
            IDs.extend(ID)

        ### Retrieve the logits and groundtruth labels for all epochs and samples
        predsDict = self.sampledict['epochlogits']
        labelsDict = self.sampledict['labels']

        loss = nn.CrossEntropyLoss()
        metricDict = {}

        ### Determine starting epochs to calculate metric over
        if self.args.epoch_loss_epochs == 0:
            startEpoch = 0
        else:
            startEpoch = self.args.tau - self.args.epoch_loss_epochs


        for ID in range(len(list(labelsDict.keys()))):
            metricDict[ID] = []
            cumulativePerformance, totalCount = 0, 0


            ### For each epoch of training up through epoch tau, record the cumulative performance on each training sample
            for e in range(startEpoch, self.args.tau, self.args.epoch_loss_interval):
                if self.args.epoch_loss_metric == "loss":
                    cumulativePerformance += (loss(predsDict[ID][e], labelsDict[ID]))
                else:
                    if torch.argmax(predsDict[ID][e]).item() == labelsDict[ID].item():
                        cumulativePerformance += 1

                totalCount += 1
                metricDict[ID].append((cumulativePerformance/totalCount))





        finalMetric = []
        reportingEpoch = self.args.tau - 1

        ### Get the cumulative metric value at the final epoch of step 1
        for ID in range(len(list(metricDict.keys()))):
            finalMetric.append(metricDict[ID][-1])

        print("Length of finalMetric: ", len(finalMetric))
    
        finalMetric = torch.tensor(finalMetric)



        #!# Ascending works best for Softmax removal, Descending works best for Loss removal (as we want to remove low accuracy or high loss samples)
        if self.args.sort_order == "ascending":
            sorted_indices = torch.argsort(finalMetric)
        elif self.args.sort_order == "descending":
            sorted_indices = torch.argsort(finalMetric, descending=True)

        ### These are the training sample indices to be removed based on the tracked metric (cumulative loss or accuracy) over the e<tau epochs of training
        sorted_indices = sorted_indices.cpu().numpy()


        ### Do class-balanced removal up to total allowed removal limit
        ### Constructs a mask of which samples to remove
        mask = []
        removedCount = 0
        class_removed = {label: 0 for label in np.unique(labels)}  # Dictionary to count class frequencies
        for idx in sorted_indices:
            ### If we can still remove more of the given class, and havent removed the total amount allowed, then remove the idx
            if (class_removed[labels[idx]] < self.args.class_removal_allowance) and (removedCount < self.args.samples_to_remove):
                class_removed[labels[idx]] += 1
                removedCount += 1
                mask.extend([idx])
                self.sampledict['removed'][IDs[idx].item()] = 1


       
        ### Now create the complement of the mask
        total_indices = set(range(len(list(metricDict.keys()))))  # Full set of indices
        mask_set = set(mask)  # Convert mask to set
        print('\n mask len is', len(mask), 'set mask len is', len(mask_set))
        mask = list(total_indices - mask_set)  # Find the complement
        return mask





    