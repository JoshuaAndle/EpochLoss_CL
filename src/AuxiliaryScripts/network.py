
import sys
import copy
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms, models


from AuxiliaryScripts import clmodels
from AuxiliaryScripts import utils

"""
The Network class is responsible for low-level functions which manipulate the model, such as training, evaluating, or selecting the classifier layer
"""
class Network():
    def __init__(self, args:argparse.Namespace, pretrained:bool="True"):

        self.args = args
        self.arch = args.arch
        self.cuda = args.cuda

        self.preprocess = None
        self.model = None
        ### The actual loading of weights is done in main()
        self.pretrained = pretrained

        self.datasets, self.classifiers = [], nn.ModuleList()

        if self.arch == "resnet18":
            self.model = clmodels.modifiedresnet18(affine=True)

        elif self.arch == "vgg16":
            self.model = clmodels.vgg16(dropout = self.args.dropout_factor)    

        else:
            sys.exit("Wrong architecture")

        if self.cuda:
            self.model = self.model.cuda()
    
        """
            When to use the backup statedict:
                Update it at the end of a task, once all the weights for the task are finalized. 
                    Use get_trainable_mask() to overwrite only indices of the newly trained/frozen weights. No other weights need be updated since they were frozen
                Read it at the start of a task, resetting the network masking to include all previously frozen weights.
                Note:
                    There should be no reason to need to load the full network backup midtask, which would be equivalent to essentially undoing the sharing/omitting decision
        """
        
        self.backupmodel = copy.deepcopy(self.model).cuda()
        ### This is a very inefficient way to reinitialize pruned weights. It can almost certainly be done without storing a full copy of the model, 
        ###   but this is simpler for the purpose of this work. At the start of a new task all newly trainable weights will be reloaded from this statedict
        self.initialmodel = copy.deepcopy(self.model).cuda()
    





    ### Add a new classifier layer for a given task
    def add_dataset(self, dataset:str, num_classes:int):
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            
            if self.arch == 'resnet18':
                self.classifiers.append(nn.Linear(512, num_classes))
            elif self.arch == 'vgg16':
                self.classifiers.append(nn.Linear(512, num_classes))


    ### Set the networks classifier layer to one of the available tasks'
    def set_dataset(self, dataset:str):
        assert dataset in self.datasets, print(f"Dataset {dataset} not found in classifiers")
        self.model.classifier = self.classifiers[self.datasets.index(dataset)]
        self.backupmodel.classifier = self.classifiers[self.datasets.index(dataset)]





    
        
        
        
    
    ### Set all frozen and pruned weights' gradients to zero for training
    def make_grads_zero(self, all_task_masks:torch.Tensor, task_num:int):
        for module_idx, (name,module) in enumerate(self.model.named_modules()):
            # print('\n name , module', name, module)
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) and name != "classifier":
                trainable_mask = utils.get_trainable_mask(module_idx, all_task_masks, task_num)
                
                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    module.weight.grad.data[trainable_mask.eq(0)] = 0
                if task_num>0 and module.bias is not None:
                    module.bias.grad.data.fill_(0)

        ### Zero gradients of all weights in the classifier connected to pruned or omitted filters
        self.model.classifier.weight.grad.data[all_task_masks[task_num]["classifier"].eq(0)] = 0


    ### Applies appropriate mask to recreate task model for inference
    def apply_mask(self, all_task_masks:torch.Tensor, task_num:int):
        """To be done to retrieve weights just for a particular dataset"""
        for module_idx, (name,module) in enumerate(self.model.named_modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) and name != "classifier":
                module.weight.data[all_task_masks[task_num][module_idx].eq(0)] = 0.0



    ### Just checks how many parameters per layer are now 0 post-pruning
    def check(self, verbose:bool=False):
        """Makes sure that the layers are pruned."""
        print('Checking...')
        for module_idx, (name,module) in enumerate(self.model.named_modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) and name != "classifier":

                weight = module.weight.data
                num_params = weight.numel()
                num_zero = weight.view(-1).eq(0).sum()
                #!# checking for which filters have been entirely pruned
                if len(weight.shape) > 2:
                    filter_mask = torch.abs(weight).le(0.000001).all(dim=3).all(dim=2).all(dim=1)
                else:
                    filter_mask = torch.abs(weight).le(0.000001).all(dim=1)
                
                num_filters = filter_mask.numel()
                num_pruned_filters = filter_mask.view(-1).sum()


                if verbose:
                    print('Layer #%d: Pruned Weights %d/%d (%.2f%%), Pruned Filters %d/%d (%.2f%%)' %
                          (module_idx, num_zero, num_params, 100 * num_zero / num_params, num_pruned_filters, num_filters, 100 * num_pruned_filters / num_filters))





    def check_weights(self):
        for name, param in self.model.named_parameters():
        # Check if the parameter is a weight
            if 'weight' in name:
                # Calculate sum or mean
                weights_sum = torch.sum(param.data)
                weights_mean = torch.mean(param.data)
                print(f"Layer: {name}, Sum of Weights: {weights_sum}, Mean of Weights: {weights_mean}")

    def check_weights_mask(self, all_task_masks:torch.Tensor, task:int=0):
        for module_idx, (name,module) in enumerate(self.model.named_modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) and name != "classifier":
                weights_sum = module.weight.data[all_task_masks[task][module_idx].eq(1)].sum()
                weights_mean = module.weight.data[all_task_masks[task][module_idx].eq(1)].mean()
                print(f"Layer: {name}, Sum of Weights: {weights_sum}, Mean of Weights: {weights_mean}")


