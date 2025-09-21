"""
Evaluates trained network on given task and setup

"""

from __future__ import division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import warnings
import copy
import random
import time

import torch
import torch.nn as nn

import numpy as np

from AuxiliaryScripts import utils, cldatasets, corruptions
from AuxiliaryScripts.manager import Manager
from AuxiliaryScripts.RemovalMetrics.Caper.Caper import Caper_Method
from AuxiliaryScripts.RemovalMetrics import EpochLoss



# To prevent PIL warnings.
warnings.filterwarnings("ignore")


###General flags
FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--run_id', type=str, default="000", help='Id of current run')
FLAGS.add_argument('--arch', choices=['resnet18', 'vgg16'], default='resnet18', help='Architectures')
FLAGS.add_argument('--pretrained', action='store_true', default=False, help='Whether or not to load a pretrained state dict prior to first task')
FLAGS.add_argument('--task_num', type=int, default=0, help='Current task number')
FLAGS.add_argument('--load_from', choices=['baseline', 'steps'], default='baseline', 
                                    help='Whether to load from baseline (no removal) or steps (removal metrics used in previous task)')

FLAGS.add_argument('--dataset', type=str, choices=['MPC', 'SynthDisjoint', 'SynthDisjoint_Reverse',
                                                "ADM", "BigGAN", "Midjourney", "glide", "stable_diffusion_v_1_4", "VQDM"], default='splitcifar', help='Name of dataset')

FLAGS.add_argument('--modifier_string', type=str, default='None,None,None,None,None,None', help='Which modifiers to use for each task in dataset')

FLAGS.add_argument('--preprocess', choices=['Normalized', 'Unnormalized'], default='Unnormalized', 
                                    help='Determines if the data is ranged 0:1 (unnormalized) or not (normalized')

FLAGS.add_argument('--attack_type', choices=['None', 'PGD', 'AutoAttack', 'gaussian_noise', 'impulse_noise', 'gaussian_blur', 'spatter', 'saturate', 'rotate'], 
                                    default='PGD', help='What type of perturbation is applied to images')
FLAGS.add_argument('--removal_metric',  type=str , default='Random', choices=['Caper', 'Random', 'NoRemoval', 'EpochLoss'], help='which metric to use for removing training samples')
FLAGS.add_argument('--trial_num', type=int , default=1, help='Trial number for setting manual seed')



# Training options.
FLAGS.add_argument('--use_train_scheduler', action='store_true', default=False, 
                                            help='If true will train with a fixed lr schedule rather than early stopping and lr decay based on validation accuracy.')
FLAGS.add_argument('--train_epochs', type=int, default=2, help='Number of epochs to train for')
FLAGS.add_argument('--eval_interval', type=int, default=5, help='The number of training epochs between evaluating accuracy')
FLAGS.add_argument('--batch_size', type=int, default=128, help='Batch size')
FLAGS.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
FLAGS.add_argument('--lr_min', type=float, default=0.001, help='Minimum learning rate below which training is stopped early')
FLAGS.add_argument('--lr_patience', type=int, default=5, help='Patience term to dictate when learning rate is decreased during training if using early stopping')
FLAGS.add_argument('--lr_factor', type=float, default=0.1, help='Multiplicative factor by which to reduce learning rate during training')

# Pruning options.
FLAGS.add_argument('--prune_perc_per_layer', type=float, default=0.65, help='Percent of weights to prune per layer')
FLAGS.add_argument('--finetune_epochs', type=int, default=2, help='Number of epochs to finetune for after pruning')




### Data Removal Options
FLAGS.add_argument('--tau',     type=int,   default=50, help='Tau determines number of epochs used for Step 1')
FLAGS.add_argument('--sort_order', choices=['ascending', 'descending'], default='descending', help='dictates sort order for various removal methods')
FLAGS.add_argument('--removal_percentage',   type=float, default=0.0, help="Percent of training samples to remove during Step 2")
# Caper-specific Options
FLAGS.add_argument('--caper_epsilon',       type=float, default=0.)
FLAGS.add_argument('--Window',              type=str, choices=['final', 'fhalf', 'shalf', 'gaussian'],   default='final', help="Which part of network to base removal decision off of")
# EpochLoss Options
FLAGS.add_argument('--epoch_loss_metric', choices=['loss', 'softmax'], default='softmax', help='How to assess performance on training data for EpochLoss removal method')
FLAGS.add_argument('--epoch_loss_epochs', type=int, default=0, help='Consider first N epochs when calculating metric')
FLAGS.add_argument('--epoch_loss_interval', type=int, default=1, help='Consider given metric averaged for every Nth epoch')


### Generally unchanged hyperparameters
FLAGS.add_argument('--cuda', action='store_true', default=True, help='use CUDA for GPU training')
FLAGS.add_argument('--save_prefix', type=str, default='./checkpoints/', help='Location to save model to')
FLAGS.add_argument('--dropout_factor', type=float, default=0.5, help='Factor for dropout layers in vgg16')
FLAGS.add_argument('--steps', choices=['step1', 'step2', 'step3', 'allsteps'], default='allsteps', 
                                help='Which steps to run. Step 1 trains network to epoch tau and gather metrics. Step 2 removes data, Step 3 resets task and trains on remaining samples.')




### Arguments dictating what task setup we are evaluating on
FLAGS.add_argument('--eval_tasknum', type=int, default=-1, help='Which task to evaluate on (should be <= args.task_num)')
FLAGS.add_argument('--eval_modifier', type=str, default='None', choices=['None', 'ai', 'nature'], help='Which modifiers to use for each tasks datasets')
FLAGS.add_argument('--eval_attack_type', choices=['None', 'PGD', 'AutoAttack', 'gaussian_noise', 'impulse_noise', 'gaussian_blur', 'spatter', 'saturate', 'rotate'], 
                                            default='PGD', help='What type of perturbation is applied')












#***# Save structure: Runid is the experiment, different task orders are subdirs that share up to the last common task so that the task can be reused/located just by giving the runid and task sequence and can be shared between multiple alternative orders for efficiency
###    Basically this just means 6 nested directories, which are nested in order of task order for the given experiment. So all subdirs of the outermost directory 2 have task 2 as the first task and can share the final dict from task 2 amongst eachother for consistency and efficiency
def main():
    args = FLAGS.parse_args()
   

    args.modifier_list = args.modifier_string.split(',')
    random.seed(args.trial_num)
    np.random.seed(args.trial_num)
    torch.manual_seed(args.trial_num)
    torch.cuda.manual_seed(args.trial_num)
    torch.cuda.manual_seed_all(args.trial_num)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)

    if args.eval_tasknum == -1:
        args.eval_tasknum = args.task_num

    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('-'*100, flush=True)   

    num_classes_by_task = utils.get_numclasses(args.dataset)

    ### Check early termination conditions
    utils.early_termination_check(args)


    taskid = args.eval_tasknum


    ######################################
    ##### Prepare Checkpoint and Manager
    ######################################

    ### Load the previous checkpoint if there is one and set the save path
    args.save_prefix, loadpath = utils.load_task_paths(args)
    loadpath, ckpt = utils.load_task_checkpoint(args, loadpath)

    manager = Manager(args, ckpt, first_task_classnum=num_classes_by_task[taskid])
    
    manager.task_num = args.eval_tasknum






    ######################################
    ##### Setup task and data
    ######################################

    ### This is for producing and setting the classifier layer for a given task's # classes
    manager.network.set_dataset(str(taskid))

    
    if args.cuda:
        manager.network.model = manager.network.model.cuda()




    ### Now that the loading is done, change the attack type to the one used for evaluation
    args.attack_type = args.eval_attack_type
    manager.args.attack_type = args.eval_attack_type

    test_data_loader = utils.get_dataloader(
                                        args.dataset, modifier=args.eval_modifier, batch_size=args.batch_size, 
                                        pin_memory=args.cuda, task_num=taskid, 
                                        set="test", preprocess=args.preprocess   
                                    )


    if args.attack_type in ['PGD', 'AutoAttack', 'None']:

        test_errors, test_errors_attacked = manager.eval(data=test_data_loader, num_class=num_classes_by_task[taskid],  use_attack=True)
        test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.
        test_accuracy_attacked = 100 - test_errors_attacked[0]  # Top-1 accuracy.


        accsPrint = [test_accuracy, test_accuracy_attacked]
        print('Final Test Vanilla and Attacked Accuracy: ', accsPrint)

    elif args.attack_type in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate']:

        accsList = {}
        test_errors = manager.eval(data=test_data_loader, num_class=num_classes_by_task[taskid],  use_attack=False)
        test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.

        accsList['Normal'] = test_accuracy

        for attack in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate']:
            test_data_loader = utils.get_dataloader(
                                                args.dataset, modifier=args.eval_modifier, batch_size=args.batch_size, 
                                                pin_memory=args.cuda, task_num=taskid, 
                                                set="test", preprocess=args.preprocess, 
                                                attack_type=attack
                                            )
            test_errors = manager.eval(data=test_data_loader, num_class=num_classes_by_task[taskid],  use_attack=False)
            test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.      

            accsList[attack] = test_accuracy

        print("All accs: ", accsList)
        print("All accs values: ", list(accsList.values()))







    return 0



    
    
if __name__ == '__main__':
    
    main()

