"""Contains utility functions primarily for data handling and removal"""
import os
import time
import copy
import math
import random 
import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from AuxiliaryScripts import DataGenerator as DG
from AuxiliaryScripts import cldatasets
from AuxiliaryScripts import manager
from AuxiliaryScripts import network as net







#####################################################
###    Misc Function
#####################################################



### Saves a checkpoint of the model
def save_ckpt(manager: manager.Manager, savename: str):
    """Saves model to file."""

    # Prepare the ckpt.
    ckpt = {
        'args': manager.args,
        'all_task_masks': manager.all_task_masks,
        'network': manager.network,
    }

    print("Saving checkpoint to ", savename)
    # Save to file.
    torch.save(ckpt, savename)




def early_termination_check(args: argparse.Namespace):

    num_classes_by_task = get_numclasses(args.dataset)

    assert args.task_num >= 0 and args.task_num < len(num_classes_by_task), f"Task num is {args.task_num}, must in range [0:{len(num_classes_by_task)}]"


    if hasattr(args, "modifier_list"):
        assert len(args.modifier_list) == len(num_classes_by_task), f"Modifier string {args.modifier_string} must have one value for each task in dataset"

    assert args.train_epochs > 0, f"Training epochs must be greater than zero. Value given: {args.train_epochs}"
    assert args.finetune_epochs >= 0, f"Finetuning epochs must have non-negative value. Value given: {args.finetune_epochs}"
    assert args.eval_interval > 0, f"Eval interval must be greater than zero. Value given: {args.eval_interval}"

    assert args.batch_size > 0, f"Batch size must be greater than zero. Value given: {args.batch_size}"
    assert args.lr > 0, f"Learning rate must be greater than zero. Value given: {args.lr}"
    assert args.lr_min > 0, f"Learning rate minimum must be greater than zero. Value given: {args.lr_min}"
    assert args.lr_patience > 0, f"Learning rate patience must be greater than zero. Value given: {args.lr_patience}"
    assert args.lr_factor > 0 and args.lr_factor < 1, f"Learning rate factor must be between 0 and 1. Value given: {args.lr_factor}"
    assert args.prune_perc_per_layer > 0 and args.prune_perc_per_layer < 1, f"Prune percent per layer must be between 0 and 1. Value given: {args.prune_perc_per_layer}"

    assert args.removal_percentage >= 0.0, f"Removal Percentage must be non-negative. Value given: {args.removal_percentage}"
    assert args.tau >= 0, f"Tau must be non-negative. Value given: {args.tau}"

    assert args.caper_epsilon >= 0.0, f"Caper Epsilon must be non-negative. Value given: {args.caper_epsilon}"

    assert args.epoch_loss_epochs >= 0, f"Epoch loss epochs must be non-negative. Value given: {args.epoch_loss_epochs}"
    assert args.epoch_loss_interval >= 1, f"Epoch loss interval must be greater than zero. Value given: {args.epoch_loss_interval}"

    assert args.dropout_factor >= 0.0 and args.dropout_factor < 1, f"Dropout factor must be in range [0.0:1.0]. Value given: {args.dropout_factor}"

    if hasattr(args, "eval_tasknum"):
        assert args.eval_tasknum >= 0 and args.eval_tasknum < len(num_classes_by_task), f"Eval task num is {args.eval_tasknum}, must in range [0:{len(num_classes_by_task)}]"




#####################################################
###    Masking Functions
#####################################################

### Get a binary mask where all previously frozen weights are indicated by a value of 1
### After pruning on the current task, this will still return the same masks, as the new weights aren't frozen until the task ends
def get_frozen_mask(weights:torch.Tensor , module_idx:int, all_task_masks:torch.Tensor, task_num:int):
    mask = torch.zeros(weights.shape)
    ### Include all weights used in past tasks (which would have been subsequently frozen)
    for i in range(0, task_num):
        if i == 0:
            mask = all_task_masks[i][module_idx].clone().detach()
        else:
            mask = torch.maximum(all_task_masks[i][module_idx], mask)
    return mask
        
    
### Get a binary mask where all unpruned, unfrozen weights are indicated by a value of 1
### Unlike get_frozen_mask(), this mask will change after pruning since the pruned weights are no longer trainable for the current task
def get_trainable_mask(module_idx:int, all_task_masks:torch.Tensor, task_num:int):
    mask = all_task_masks[task_num][module_idx].clone().detach()
    frozen_mask = get_frozen_mask(mask, module_idx, all_task_masks, task_num)
    mask[frozen_mask.eq(1)] = 0
    return mask
    

        




#####################################################
###    Dataset Functions
#####################################################


### Number of classes by task
def get_numclasses(dataset:str):
    if dataset == 'MPC':
        numclasses = [20,10,20,10,20,10]
    elif dataset in ['SynthDisjoint', "SynthDisjoint_Reverse", "ADM", "BigGAN", "Midjourney", "glide", "stable_diffusion_v_1_4", "VQDM"]:
        numclasses = [100,100,100,100,100,100]
    
    return numclasses
    
### Returns a dictionary of "train", "valid", and "test" data+labels for the appropriate cifar subset
def get_dataloader(
    dataset:str, modifier:str, batch_size:int, num_workers:int=4, 
    pin_memory:bool=False, task_num:int=0, set:str="train", 
    preprocess:str="Normalized", shuffle:bool=False, attack_type:Optional[str]=None
    ):

    # standard split CIFAR-10/100 sequence of tasks

    if dataset == "MPC":
        dataset = cldatasets.get_mixedCIFAR_PMNIST(task_num=task_num, split = set, preprocess=preprocess, attack=attack_type)
    elif dataset == "SynthDisjoint":
        dataset = cldatasets.get_Synthetic(task_num=task_num, split = set, modifier=modifier, preprocess=preprocess, attack=attack_type)
    elif dataset == "SynthDisjoint_Reverse":
        dataset = cldatasets.get_Synthetic(task_num=task_num, split = set, modifier=modifier, preprocess=preprocess, attack=attack_type, order="reversed")
    elif dataset in ["ADM", "BigGAN", "Midjourney", "glide", "stable_diffusion_v_1_4", "VQDM"]:
        dataset = cldatasets.get_Synthetic_SingleGenerator(task_num=task_num, split = set, generator = dataset, modifier=modifier, preprocess=preprocess, attack=attack_type)



    else: 
        print("Incorrect dataset for get_dataloader()")
        return -1
        

    
    IDs = torch.arange(len(dataset['y']))

    ### Makes a custom dataset for a given dataset through torch
    # generator = DG.SimpleDataGenerator(dataset['x'],dataset['y'])
    generator = DG.IdTrackDataGenerator(dataset['x'],dataset['y'], IDs)
    
    


    
    ### Loads the custom data into the dataloader
    if set == "train":        
        return data.DataLoader(generator, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory=pin_memory)
    else:
        return data.DataLoader(generator, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory=pin_memory)








### Splits training dataset into tuples of (x,y,z) for later data removal
def prepare_allbatches(dataset:dict):        
    new_indices = torch.arange(len(dataset['y']))

    # x_unbound = torch.unbind(dataset['x'])
    # print("Len of x_unbound: ", len(x_unbound))
    # print("Shape of element in x_unbound: ", x_unbound[0].shape)

    all_batches = list(zip(
        torch.unbind(dataset['x']), 
        torch.unbind(dataset['y']), 
        torch.unbind(dataset['z'])
        ))   


    total_sum = 0
    labelcount = {}
    for _, label, _ in all_batches:
        if label.item() in labelcount.keys():
            labelcount[label.item()] += 1
        else:
            labelcount[label.item()] = 1
        total_sum += 1

    print('\nOriginal total number of samples is', total_sum)
    print("Original number of samples by label: ",labelcount)

    return all_batches







### Get the paths for loading and saving the current task
def load_task_paths(args:argparse.Namespace):

    ### Since autoattack is only used for evaluation, dont want to have to rerun baselines separately for autoattack and PGD
    ### To avoid this, I've set it up to use a shared load path but save to different paths based on attack type    

    if args.use_train_scheduler==True:
        loadpath = os.path.join("./checkpoints/", str(args.dataset) + "_" + str(args.arch), str(args.run_id), 'trial-'+ str(args.trial_num),
                                    str(args.attack_type), str(args.prune_perc_per_layer),
                                    'epochs-'+ str(args.train_epochs) + "_batch_size-" + str(args.batch_size), 'using_scheduler')
    else:
        loadpath= os.path.join("./checkpoints/", str(args.dataset) + "_" + str(args.arch), str(args.run_id), 'trial-'+ str(args.trial_num),
                                    str(args.attack_type), str(args.prune_perc_per_layer),
                                    'epochs-'+ str(args.train_epochs) + "_batch_size-" + str(args.batch_size), 
                                    'lr-'+str(args.lr) + '_patience-'+str(args.lr_patience) + '_factor-'+str(args.lr_factor) + '_lrmin-'+str(args.lr_min))
    
    savepath = loadpath


    ### All save paths start in a directory based on the shared hyperparameter values for that metric
    if args.removal_metric in ['Caper']:
        savepath = os.path.join(savepath, 'tau-'+ str(args.tau),  'metric-' + str(args.removal_metric), 
                                'sample_percent-'+ str(args.removal_percentage * 100) + "_sorting-" + args.sort_order)

    elif args.removal_metric in ['EpochAcc']:
        savepath = os.path.join(savepath, 'tau-'+ str(args.tau),  'metric-' + str(args.removal_metric),
                                'sample_percent-'+ str(args.removal_percentage * 100) + "_sorting-" + args.sort_order,
                                'EpochAccMetric-'+ str(args.EpochAccMetric) + '_NumEpochs-' + str(args.EpochAccEpochs) + '_Interval-' + str(args.EpochAccInterval)) 
                                 
    elif args.removal_metric in ['Random']:
        savepath = os.path.join(savepath, 'tau-'+ str(args.tau),  'metric-' + str(args.removal_metric), 
                                'sample_percent-'+ str(args.removal_percentage * 100))




    ### Load from within the nested removal directories
    #!# If loading from removal metric, need the loadpath to reflect the appropriate hyperparameter subdirectories
    if args.load_from != "baseline":
        loadpath = savepath



    ### Update loadpath for each previous task in the current sequence
    for t in range(0, args.task_num):
        ### Setup the appropriate modifier string for the given task. If not doing synthetic data, omit modifier for simplicity
        if args.modifier_list[t] == "None":
            task_modifier = ""
        else:
            task_modifier = (args.modifier_list[t] + "_")

        if args.load_from == "baseline":
            loadpath = os.path.join(loadpath, (task_modifier + str(t) + "_" + "NoRemoval"))
            savepath = os.path.join(savepath, (task_modifier + str(t) + "_" + "NoRemoval"))
        else:
            ### Note: baseline is saved as "NoRemoval" metric
            loadpath = os.path.join(loadpath, (task_modifier + str(t) + "_" + args.removal_metric))
            savepath = os.path.join(savepath, (task_modifier + str(t) + "_" + args.removal_metric))


    ### Extend the savepath from the loadpath to include the current task
    if args.modifier_list[args.task_num] == "None":
        task_modifier = ""
    else:
        task_modifier = args.modifier_list[args.task_num] + "_"


    ### Note: baseline is saved as "NoRemoval" metric
    savepath = os.path.join(savepath, (task_modifier + str(args.task_num) + "_" + args.removal_metric))

    return savepath, loadpath



### Load checkpoint if available given a loadpath
def load_task_checkpoint(args:argparse.Namespace, loadpath:str):
    ckpt = None

    ### If no checkpoint is found, the default value will be None and a new one will be initialized in the Manager
    if args.task_num != 0:
        ### Path to load previous task's checkpoint, if not starting at task 0
        previous_task_path = os.path.join(loadpath, "final.pt") 
        print('path is', previous_task_path)
        ### Reloads checkpoint depending on where you are at for the current task's progress (t->c->p)    
        if os.path.isfile(previous_task_path) == True:
            ckpt = torch.load(previous_task_path)
            print("Checkpoint found and loaded from: ", previous_task_path)
        else:
            print("!!!No checkpoint file found at ", previous_task_path)
            raise FileNotFoundError

    return loadpath, ckpt







### Load pretrained weights for first task
def load_pretrained(args:argparse.Namespace, manager:manager.Manager):
    if args.arch == "vgg16":
        pretrained_state_dict=torch.load('pretrained_model_weights_vgg.pt')
       
        model_state_dict = manager.network.model.state_dict()
        for name, param in pretrained_state_dict.items():
            if 'avgpool' in name:
                continue  # Skip the average pooling layer
            elif 'classifier1' in name and 'weight' in name:
                name='features.45.weight'
            elif 'classifier1' in name and 'bias' in name:
                name='features.45.bias'
            elif 'features' in name:
                l = name.split('.')
                name = l[0]+'.'+l[1].split('_')[1]+'.'+ l[2]

            if name in model_state_dict:
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name].copy_(param)
                else:
                    print(f"Shape mismatch for layer {name}, skipping this layer.")
            else:
                print(f"Layer {name} not found in custom model, skipping this layer.")
    elif args.arch == "resnet18":
        pretrained_state_dict=torch.load('pretrained_model_weights_mrn18_affine.pt')
        model_state_dict = {k: v for k, v in pretrained_state_dict.items() if 'classifier' not in k}
    
    return model_state_dict










#####################################################
###    Data Removal Functions
#####################################################

### Remove a random subset of training samples and return a resulting dataloader and the removed indices
def random_remove(all_batches:list, number_to_remove:int, batch_size:int, cuda:bool=True):
    sets_to_remove = random.sample(range(len(all_batches)), number_to_remove)
    indices_to_keep = [i for i in range(len(all_batches)) if i not in sets_to_remove]

    x_batches = [all_batches[i][0] for i in indices_to_keep]
    y_batches = [all_batches[i][1] for i in indices_to_keep]
    z_batches = [all_batches[i][2] for i in indices_to_keep]



    # Concatenate along the batch dimension
    x_concatenated = torch.stack(x_batches, dim=0)
    print("Shape of x_concatenated: ", x_concatenated.shape)
    y_concatenated = torch.stack(y_batches, dim=0)
    z_concatenated = torch.stack(z_batches, dim=0)

    # Create a new dataset with the concatenated batches
    train_new_data_loader = list(zip(x_concatenated, y_concatenated, z_concatenated))
    train_new_data_loader = data.DataLoader(train_new_data_loader, batch_size= batch_size, shuffle = True, num_workers = 4, pin_memory=cuda)
    
    return train_new_data_loader, sets_to_remove



### Remove a subset of training samples based on mask produced by removal metric. Returns a resulting dataloader and the removed indices
def remove_masked_samples(all_batches:list, sample_mask:list[int], batch_size:int, cuda:bool=True):
    x_batches = [all_batches[i][0] for i in sample_mask]
    # print("Shape of x_concatenated: ", x_concatenated.shape)
    y_batches = [all_batches[i][1] for i in sample_mask]
    z_batches = [all_batches[i][2] for i in sample_mask]

     # Concatenate along the batch dimension
    x_concatenated = torch.stack(x_batches, dim=0)
    y_concatenated = torch.stack(y_batches, dim=0)
    z_concatenated = torch.stack(z_batches, dim=0)
    # print("Remaining caper IDs: ", z_concatenated)
    # Create a new dataset with the concatenated batches
    train_new_data_loader = list(zip(x_concatenated, y_concatenated, z_concatenated))
    train_new_data_loader = data.DataLoader(train_new_data_loader, batch_size= batch_size, shuffle = True, num_workers = 4, pin_memory=cuda)
    
    return train_new_data_loader




