"""
Generates and saves corrupted test dataset split for specified task
Must be run prior to training and evaluating on corruption data, since pre-generated test sets are used for consistency

"""

from __future__ import division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import warnings
import copy
import time
import random

import numpy as np
import torch
import torch.nn as nn

from AuxiliaryScripts import utils, cldatasets, corruptions
from AuxiliaryScripts.manager import Manager
from AuxiliaryScripts.RemovalMetrics.Caper.Caper import Caper_Method
from AuxiliaryScripts.RemovalMetrics import EpochLoss


# To prevent PIL warnings.
warnings.filterwarnings("ignore")

###General flags
FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--run_id', type=str, default="000", help='Id of current run.')
FLAGS.add_argument('--arch', choices=['resnet18', 'vgg16'], default='resnet18', help='Architectures')
FLAGS.add_argument('--pretrained', action='store_true', default=False, help='Whether or not to load a predefined pretrained state dict in Network().')
FLAGS.add_argument('--load_from', choices=['baseline', 'steps'], default='baseline', help='Whether or not we are loading from the baseline')
FLAGS.add_argument('--task_num', type=int, default=0, help='Current task number.')

FLAGS.add_argument('--dataset', type=str, choices=['MPC', 'SynthDisjoint'], default='splitcifar', help='Name of dataset')
FLAGS.add_argument('--dataset_modifier', choices=['None', 'ai', 'nature'], default='None', help='Determines which variant of certain datasets is used')
FLAGS.add_argument('--preprocess', choices=['Normalized', 'Unnormalized'], default='Unnormalized', help='Determines if the data is ranged 0:1 unnormalized or not (normalized')

FLAGS.add_argument('--attack_type', choices=['PGD', 'AutoAttack', 'gaussian_noise', 'impulse_noise', 'gaussian_blur', 'spatter', 'saturate', 'rotate'], default='PGD', help='What type of perturbation is applied')
FLAGS.add_argument('--removal_metric',  type=str , default='Random', choices=['Caper', 'Random', 'NoRemoval', 'EpochLoss'], help='which metric to use for removing training samples')
FLAGS.add_argument('--trial_num', type=int , default=1, help='Trial number for setting manual seed')



# Training options.
FLAGS.add_argument('--use_train_scheduler', action='store_true', default=False, help='If true will train with a fixed lr schedule rather than early stopping and lr decay based on validation accuracy.')
FLAGS.add_argument('--train_epochs', type=int, default=2, help='Number of epochs to train for')
FLAGS.add_argument('--eval_interval', type=int, default=5, help='The number of training epochs between evaluating accuracy')
FLAGS.add_argument('--batch_size', type=int, default=128, help='Batch size')
FLAGS.add_argument('--lr', type=float, default=0.1, help='Learning rate')
FLAGS.add_argument('--lr_min', type=float, default=0.001, help='Minimum learning rate below which training is stopped early')
FLAGS.add_argument('--lr_patience', type=int, default=5, help='Patience term to dictate when Learning rate is decreased during training')
FLAGS.add_argument('--lr_factor', type=float, default=0.1, help='Factor by which to reduce learning rate during training')

# Pruning options.
### Note: We only use structured pruning for now. May try unstructured pruning as well unless it causes issues with CL weight sharing, but it likely shouldnt. 
FLAGS.add_argument('--prune_perc_per_layer', type=float, default=0.65, help='% of neurons to prune per layer')
FLAGS.add_argument('--finetune_epochs', type=int, default=2, help='Number of epochs to finetune for after pruning')




### Data Removal Options
FLAGS.add_argument('--tau',     type=int,   default=50, help='Tau')
FLAGS.add_argument('--sort_order', choices=['ascending', 'descending'], default='descending', help='dictates sort order for various removal methods')
# Caper-specific Options
FLAGS.add_argument('--caper_epsilon',       type=float, default=0.)
FLAGS.add_argument('--Window',              type=str, choices=['final', 'fhalf', 'shalf', 'gaussian'],   default='final')
FLAGS.add_argument('--removal_percentage',   type=float, default=0.0)
FLAGS.add_argument('--class_removal_allowance', type=int ,  default=100)

# EpochLoss Options
FLAGS.add_argument('--epoch_loss_metric', choices=['loss', 'softmax'], default='softmax', help='How to assess performance on training data for EpochLoss removal method')


### Generally unchanged hyperparameters
FLAGS.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
FLAGS.add_argument('--save_prefix', type=str, default='./checkpoints/', help='Location to save model')
FLAGS.add_argument('--steps', choices=['step1', 'step2', 'step3', 'allsteps'], default='allsteps', help='Which steps to run')
FLAGS.add_argument('--dropout_factor', type=float, default=0.5, help='Factor for dropout layers in vgg16')





def main():
    args = FLAGS.parse_args()
   
    random.seed(args.trial_num)
    np.random.seed(args.trial_num)
    torch.manual_seed(args.trial_num)
    torch.cuda.manual_seed(args.trial_num)
    torch.cuda.manual_seed_all(args.trial_num)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)


    num_classes_by_task = utils.get_numclasses(args.dataset)
    
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('-'*100, flush=True)   

    ### Check early termination conditions
    utils.early_termination_check(args)

    taskid = args.task_num


    # ###################
    # ##### Setup task and data
    # ###################



    if args.dataset == "MPC":
        dataset = cldatasets.get_mixedCIFAR_PMNIST(task_num=args.task_num, split = 'test', modifier=args.dataset_modifier, preprocess=args.preprocess)
    elif args.dataset == "SynthDisjoint":
        dataset = cldatasets.get_Synthetic(task_num=args.task_num, split = 'test', subset = 'disjoint', modifier=args.dataset_modifier, preprocess=args.preprocess)
    elif args.dataset == "SynthDisjoint_Reverse":
        dataset = cldatasets.get_Synthetic(task_num=args.task_num, split = 'test', modifier=args.dataset_modifier, preprocess=args.preprocess, order="reversed")

    elif args.dataset in ["ADM", "BigGAN", "Midjourney", "glide", "stable_diffusion_v_1_4", "VQDM"]:
        dataset = cldatasets.get_Synthetic_SingleGenerator(task_num=args.task_num, split = 'test', generator = args.dataset, modifier=args.dataset_modifier, preprocess=args.preprocess)


    images = dataset['x']
    labels = dataset['y']



    images_corrupted = copy.deepcopy(images)

    if args.attack_type == "gaussian_noise":
        print("Using attack: ", args.attack_type)
        for i in range(len(images_corrupted)):
            images_corrupted[i] = corruptions.gaussian_noise(images_corrupted[i], severity=3)

    elif args.attack_type == "gaussian_blur":
        print("Using attack: ", args.attack_type)
        for i in range(len(images_corrupted)):
            images_corrupted[i] = corruptions.gaussian_blur(images_corrupted[i], severity=3)

    elif args.attack_type == "saturate":
        print("Using attack: ", args.attack_type)
        for i in range(len(images_corrupted)):
            images_corrupted[i] = corruptions.saturate(images_corrupted[i], severity=3)

    elif args.attack_type == "rotate":
        print("Using attack: ", args.attack_type)
        for i in range(len(images_corrupted)):
            images_corrupted[i] = corruptions.rotate(images_corrupted[i], severity=3)



    print("Image Corrupted size: ", images_corrupted.size(), flush=True)


    if args.dataset == "MPC":
        if args.task_num in [1,3,5]:
            torch.save(images_corrupted, os.path.join(os.path.expanduser(('./data/PMNIST/' + str(args.task_num+1))), ('x_' + args.attack_type + '_test.bin')))

        else:
            ### Skips the first split_cifar task which is the larger CIFAR-10 dataset, only uses the smaller tasks of CIFAR-100
            if args.task_num == 0:
                args.task_num = 1
            torch.save(images_corrupted, os.path.join(os.path.expanduser(('./data/split_cifar/' + str(args.task_num))), ('x_' + args.attack_type + '_test.bin')))

    elif args.dataset == "SynthDisjoint":
        taskDict = {0:"ADM", 1:"BigGAN", 2:"Midjourney", 3:"glide", 4:"stable_diffusion_v_1_4", 5:"VQDM"}
        savepath = os.path.join(os.path.expanduser('./data/Synthetic'), 
                                taskDict[args.task_num], str(args.task_num), 'test', args.dataset_modifier, 
                                ('X_' + args.attack_type + '.pt'))
        print("Saving to: ", savepath)
        torch.save(images_corrupted, savepath)


    elif args.dataset == "SynthDisjoint_Reverse":
        taskDict = {0:"VQDM",1:"stable_diffusion_v_1_4",  2:"glide",3:"Midjourney", 4:"BigGAN",  5:"ADM"}
        savepath = os.path.join(os.path.expanduser('./data/Synthetic'), 
                                taskDict[args.task_num], str(args.task_num), 'test', args.dataset_modifier, 
                                ('X_' + args.attack_type + '.pt'))
        print("Saving to: ", savepath)
        torch.save(images_corrupted, savepath)


    elif args.dataset in ["ADM", "BigGAN", "Midjourney", "glide", "stable_diffusion_v_1_4", "VQDM"]:
        taskDict = {0:a, 1:"BigGAN", 2:"Midjourney", 3:"glide", 4:"stable_diffusion_v_1_4", 5:"VQDM"}
        savepath = os.path.join(os.path.expanduser('./data/Synthetic'), 
                                args.dataset, str(args.task_num), 'test', args.dataset_modifier, 
                                ('X_' + args.attack_type + '.pt'))
        print("Saving to: ", savepath)
        torch.save(images_corrupted, savepath)





    return 0



    
    
if __name__ == '__main__':
    
    main()

