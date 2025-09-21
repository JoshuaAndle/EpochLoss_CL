"""
Does standard subnetwork training on all tasks

"""

from __future__ import division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import warnings
import copy
import random

import torch
import torch.nn as nn

import numpy as np

from math import floor

from AuxiliaryScripts import utils, cldatasets, corruptions
from AuxiliaryScripts.manager import Manager
from AuxiliaryScripts.RemovalMetrics.Caper.Caper import Caper_Method
from AuxiliaryScripts.RemovalMetrics import EpochLoss



# To prevent PIL warnings.
warnings.filterwarnings("ignore")

###General flags
FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--run_id', type=str, default="000", help='Id of current run')
FLAGS.add_argument('--trial_num', type=int , default=1, help='Trial number for setting manual seed')
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
FLAGS.add_argument('--removal_metric',  type=str , default='Random', choices=['Caper', 'Random', 'NoRemoval', 'EpochLoss'], help='which metric to use for removing training samples')
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
FLAGS.add_argument('--steps', choices=['step1', 'step2', 'step3', 'allsteps'], default='step3', 
                                help='Which steps to run. Step 1 trains network to epoch tau and gather metrics. Step 2 removes data, Step 3 resets task and trains on remaining samples.')


### Shared function for evaluating test accuracy after training, pruning, and finetuning
def test_eval(args: argparse.Namespace, manager: Manager, num_classes: int):
    
    test_data_loader = utils.get_dataloader(
                                        args.dataset, modifier=args.modifier_list[args.task_num], batch_size=args.batch_size, 
                                        pin_memory=args.cuda, task_num=args.task_num, 
                                        set="test", preprocess=args.preprocess
                                    )

    ### Evaluate trained model on unseen test data
    if args.attack_type in ['PGD', 'AutoAttack', 'None']:
        print("Getting test accuracies for attack type: ", args.attack_type)

        test_errors, test_errors_attacked = manager.eval(data=test_data_loader, num_class=num_classes,  use_attack=True)
        test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.
        test_accuracy_attacked = 100 - test_errors_attacked[0]  # Top-1 accuracy.

        print('Final Test Vanilla and Attacked Accuracy: ', [test_accuracy, test_accuracy_attacked])




    elif args.attack_type in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate']:

        print("Getting test accuracies for normal test data")
        test_errors = manager.eval(data=test_data_loader, num_class=num_classes,  use_attack=False)
        test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.
        print('Final Test Vanilla Accuracy on un-attacked data: %0.2f%%' %(test_accuracy))

        
        accsList = {'Normal': test_accuracy}

        for attack in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate']:
            print("\nGetting test accuracies for corrupted test data with corruption: ", attack)
            test_data_loader = utils.get_dataloader(
                                                args.dataset, modifier=args.modifier_list[args.task_num], batch_size=args.batch_size, 
                                                pin_memory=args.cuda, task_num=args.task_num, set="test", preprocess=args.preprocess, 
                                                attack_type=attack
                                            )

            ### Note: use_attack is False since corruptions are pre-applied to stored test dataset split
            test_errors = manager.eval(data=test_data_loader, num_class=num_classes,  use_attack=False)
            test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.
            print('Final Test Accuracy on ', attack ,' data: %0.2f%%' %(test_accuracy))
            
            accsList[attack] = test_accuracy

            print("All accs: ", accsList)
            print("All accs values: ", list(accsList.values()))










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



    args.modifier_list = args.modifier_string.split(',')
    num_classes_by_task = utils.get_numclasses(args.dataset)

    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('-'*100, flush=True)   



    ### Check early termination conditions
    utils.early_termination_check(args)
    





    ######################################
    ##### Prepare Checkpoint and Manager
    ######################################

    ### Load the previous checkpoint if there is one and set the save path
    args.save_prefix, loadpath = utils.load_task_paths(args)
    loadpath, ckpt = utils.load_task_checkpoint(args, loadpath)

    #!# Leaving in incase we want to expand to doing alternative task orders
    taskid = args.task_num
    manager = Manager(args, ckpt, first_task_classnum=num_classes_by_task[taskid])
    
    if args.pretrained and taskid==0:
        ### Load a compatible version of the pretrained weights prior to first task
        pretrained_dict = utils.load_pretrained(args, manager)
        manager.network.model.load_state_dict(pretrained_dict, strict=False)







    ######################################
    ##### Setup task and data
    ######################################
    
    ### Update paths as needed for each new task
    print("Task ID: ", taskid, " #", args.task_num, " in sequence for dataset: ", args.dataset)
    print('\n\n args.save_prefix  is ', args.save_prefix, "\n\n", flush=True)
    os.makedirs(args.save_prefix, exist_ok = True)

    manager.save_prefix = args.save_prefix

    trained_path, finetuned_path = os.path.join(args.save_prefix, "trained.pt"), os.path.join(args.save_prefix, "final.pt") 
    print("Finetuned path: ", finetuned_path, flush=True)

    ### Prepare dataloaders for new task
    ### Note: extra_loader is a non-shuffled version of train dataloader used when removing data
    train_data_loader = utils.get_dataloader(args.dataset, modifier=args.modifier_list[args.task_num], batch_size=args.batch_size, pin_memory=args.cuda, task_num=taskid, set="train", preprocess=args.preprocess, shuffle=True)
    val_data_loader  =  utils.get_dataloader(args.dataset, modifier=args.modifier_list[args.task_num], batch_size=args.batch_size, pin_memory=args.cuda, task_num=taskid, set="valid", preprocess=args.preprocess)
    extra_data_loader =  utils.get_dataloader(args.dataset, modifier=args.modifier_list[args.task_num], batch_size=args.batch_size, pin_memory=args.cuda, task_num=taskid, set="train", preprocess=args.preprocess)
    manager.train_loader, manager.val_loader, manager.extra_loader = train_data_loader, val_data_loader, extra_data_loader

    if args.dataset == "MPC":
        dataset = cldatasets.get_mixedCIFAR_PMNIST(task_num=args.task_num, split = 'train', preprocess=args.preprocess)
    elif args.dataset == "SynthDisjoint":
        dataset = cldatasets.get_Synthetic(task_num=args.task_num, split = 'train', modifier=args.modifier_list[args.task_num], preprocess=args.preprocess)
    elif args.dataset == "SynthDisjoint_Reverse":
        dataset = cldatasets.get_Synthetic(task_num=args.task_num, split = 'train', modifier=args.modifier_list[args.task_num], preprocess=args.preprocess, order="reversed")

    elif args.dataset in ["ADM", "BigGAN", "Midjourney", "glide", "stable_diffusion_v_1_4", "VQDM"]:
        dataset = cldatasets.get_Synthetic_SingleGenerator(task_num=args.task_num, split = 'train', generator = args.dataset, modifier=args.modifier_list[args.task_num], preprocess=args.preprocess)

    ### Initialize the z (unique sample ID for tracking) values of the dataset before changing anything
    dataset['z'] = torch.arange(len(dataset['y']))

    all_batches = utils.prepare_allbatches(dataset=copy.deepcopy(dataset))



    total_train_images = sum(len(batch[0]) for batch in manager.train_loader)
    args.samples_to_remove = round((args.removal_percentage/100) * total_train_images)
    args.class_removal_allowance = floor(args.samples_to_remove / num_classes_by_task[taskid])
    print('Samples to remove: {} of {} total training samples'.format(args.samples_to_remove, total_train_images), flush=True)
    print("Class removal allowance: ", args.class_removal_allowance)

    total_extra_images = sum(len(batch[0]) for batch in manager.extra_loader)
    print('Samples in extra_loader :', total_extra_images, flush=True)


    ### This is for producing and setting the classifier layer for a given task's # classes
    manager.network.add_dataset(str(args.task_num), num_classes_by_task[taskid])
    manager.network.set_dataset(str(args.task_num))
    if args.cuda:
        manager.network.model = manager.network.model.cuda()





    ### Track the softmax of normal and attacked samples at tau, the logits of epochs in step 1, and whether the sample's removed
    sampledict = {"tau_logits":{}, "tau_advlogits":{}, "epochlogits": {}, "removed":{}, "labels":{}, 'step2_time':{}}
    for i in range(len(dataset['y'])):
        sampledict['tau_logits'][i] = []
        sampledict['tau_advlogits'][i] = []
        sampledict['epochlogits'][i] = []
        sampledict['removed'][i] = 0
        sampledict['labels'][i] = 0
        sampledict['step2_time'][i] = 0

    ### Passing this to manager so that the step 1 logits can be passed back after epoch tau
    manager.sampledict = sampledict


    
    ### Reload all previously masked weights to get the full network prior to weight sharing.
    if args.task_num != 0:
        manager.prepare_task()


    ### Copy the manager and network so we can reload after epoch tau
    manager_deep_copy = copy.deepcopy(manager)






    ##################################################################################################################
    ##### Step 1: Train non-adversarially for tau epochs
    ##################################################################################################################
    if args.removal_metric != "NoRemoval" and args.tau != 0 and args.steps in ["step1", 'allsteps']:
        print('\n\n\n', '-' * 16, '\nStep 1 is started \n', flush=True)

        trained_path = os.path.join(args.save_prefix, (args.removal_metric + "_1-2_trained.pt"))
        
        manager.train(args.tau, save=False, savename=trained_path, num_class=num_classes_by_task[taskid], use_attack=False, save_best_model=True, trackpreds=True)
        #!# Should save here, but commented out due to storage limitations and non-adversarial training steps are relatively quick
        # utils.save_ckpt(manager, savename=trained_path)


        ### Store the logits for each epoch of step 1 in the dict
        sampledict['epochlogits'] = manager.sampledict['epochlogits']
        sampledict['labels'] = manager.sampledict['labels']

        

    ##################################################################################################################
    ##### Step 2: Remove samples predicted to reduce robustness/accuracy of model based on chosen metric
    ##################################################################################################################

    ### Remove data from training dataset with given method, then return new training dataloader with only remaining samples
    if args.removal_metric not in ["NoRemoval"] and args.steps in ["step2", 'allsteps']:
        print("\n\n\n", '-' * 16, '\nstep 2 is started \n', flush=True)
        print("Removing data with metric: ", args.removal_metric, '\n\n')

        if args.removal_metric == "Random":
            print("\n\nCalculating random removal")
            train_new_data_loader, sets_to_remove_random = utils.random_remove(all_batches, args.samples_to_remove, args.batch_size)
            sets_to_remove_random = torch.tensor(sets_to_remove_random)
            torch.save(sets_to_remove_random, (args.save_prefix + "/removed_indices_random.pt"))
        
            ### Note which samples were removed
            for i in range(len(sets_to_remove_random)):
                ID = all_batches[sets_to_remove_random[i]][2]
                sampledict['removed'][ID.item()] = 1





        elif args.removal_metric == "EpochLoss":
            print("\n\nCalculating EpochLoss")
            EpochLossClass = EpochLoss.EpochLoss_Method(args, manager.network.model, extra_data_loader, sampledict=sampledict)
            sample_mask = EpochLossClass.gen_data_mask()

            train_new_data_loader = utils.remove_masked_samples(all_batches, sample_mask, args.batch_size)
            # sets_to_remove = torch.from_numpy(sample_mask)
            sets_to_remove = sample_mask
            # print("Sets to remove EpochLoss: ", sets_to_remove)
            torch.save(sets_to_remove, (args.save_prefix + "/removed_indices_epoch_loss.pt"))


            sampledict = EpochLossClass.sampledict




        elif args.removal_metric == "Caper":
            print("\n\nCalculating caper")
            CaperClass = Caper_Method(args, manager.network.model, extra_data_loader, sampledict=sampledict)
            sample_mask = CaperClass.New_Data(args.save_prefix)

            train_new_data_loader = utils.remove_masked_samples(all_batches, sample_mask, args.batch_size)
            # sets_to_remove = torch.from_numpy(sample_mask)
            sets_to_remove = sample_mask
            # print("Sets to remove caper: ", sets_to_remove)
            torch.save(sets_to_remove, (args.save_prefix + "/removed_indices_caper.pt"))


            sampledict = CaperClass.sampledict







        ### Save the sampledict now that removal is complete
        sampledict_path = os.path.join(args.save_prefix, ("sampledict.pt"))
        torch.save(sampledict, sampledict_path)

        ### Replace train_loader with new loader contianing only remaining samples

        ### Reloading checkpoint stored at start of task
        manager = manager_deep_copy
        manager.train_loader = train_new_data_loader

        # trained_path = os.path.join(args.save_prefix, ((args.removal_metric + "2-trained.pt")))
        # utils.save_ckpt(manager, savename=trained_path)

        
        total_sum = 0
        labelcount = {}
        for _, labels, _ in train_new_data_loader:
            for label in labels:
                if label.item() in labelcount.keys():
                    labelcount[label.item()] += 1
                else:
                    labelcount[label.item()] = 1
                total_sum += 1
            # total_sum += torch.sum(labels).item()
        print('Remaining number of samples in updated train_loader is ', total_sum)
        # print('Remaining samples by label: ', labelcount)
        
        sorted_by_keys = {key: labelcount[key] for key in sorted(labelcount)}
        print("Sorted remaining labels: ", sorted_by_keys)











    ##################################################################################################################
    ##### Step 3: Adversarially train to convergence and check test accuracy afterwards.
    ##################################################################################################################
    
    #*# Run step 3 if set to do so, otherwise we skip it if just interested in reporting metrics at epoch tau
    if args.steps in ["step3", 'allsteps']:
        print("\n\n\n", '-' * 16, '\nstep 3 is started \n', flush=True)
    
        num_samples = len(manager.train_loader.dataset)
        print(f"Number of samples in the new train dataLoader for step 3: {num_samples}")

        if args.cuda:
            manager.network.model = manager.network.model.cuda()
            
        trained_path = os.path.join(args.save_prefix, ((args.removal_metric + "_steps3-trained.pt")))
        manager.train(args.train_epochs, save=True, savename=trained_path, num_class=num_classes_by_task[taskid])
        # manager.train(args.train_epochs, save=True, savename=trained_path, num_class=num_classes_by_task[taskid], use_attack=False)
        utils.save_ckpt(manager, savename=trained_path)

        ### Get test accuracies
        test_eval(args, manager, num_classes_by_task[taskid])
       
        # traincurve_path = os.path.join(args.save_prefix, ("step3-trainingcurve.pt"))
        # train_curve_dict = {'valAccsNormal': val_acc_history, "valAccsAttacked": val_acc_history_attacked}
        # torch.save(train_curve_dict, traincurve_path)








    ##################################################################################################################
    ##### Prune and Finetune if continuing to next task
    ##################################################################################################################

    ### If continuing to next task, then also do pruning and finetuning
    if args.steps in ['allsteps']:

        ### Prune unecessary weights or nodes
        manager.prune()
        print('\nPost-prune eval:')

        utils.save_ckpt(manager, finetuned_path)

        ### Get test accuracies after pruning
        test_eval(args, manager, num_classes_by_task[taskid])




        if args.finetune_epochs > 0:
            print('Doing some extra finetuning...')
            manager.train(args.finetune_epochs, save=True, savename=finetuned_path, num_class=num_classes_by_task[taskid])

            ### Save the checkpoint and move on to the next task if required
            utils.save_ckpt(manager, finetuned_path)

            ### Get test accuracies after finetuning
            test_eval(args, manager, num_classes_by_task[taskid])



        print('-' * 16)
        print('Pruning summary:')
        manager.network.check(True)
        print('-' * 16)
        print("\n\n\n\n")



    return 0



    
    
if __name__ == '__main__':
    
    main()

