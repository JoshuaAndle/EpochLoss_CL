import os
import sys
import numpy as np
import copy
from typing import Optional
import random

import math
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms




### Not used in paper, but mixes tasks from CIFAR and Permuted MNIST
def get_mixedCIFAR_PMNIST(
    seed:int = 0, pc_valid:float=0.1, task_num:int=0, 
    split:str="", offset:bool=False, preprocess:str="Normalized", 
    attack:Optional[str]=None, reduced:bool=True
    ):
    """
    Sequence: 
        0:CIFAR100 split 1, 
        1:Permuted MNIST 1, 
        2:CIFAR100 split 2, 
        3:Permuted MNIST 2, 
        4:CIFAR100 split 4, 
        5:Permuted MNIST 3, 
    """
    ### Check if tasks have been generated yet. If not, makes them
    if os.path.isfile(("./data/split_cifar/" + str(task_num) + "/x_train.bin")) == False:
        print("No dataset detected for cifar subsets. Creating new set prior to loading task.")
        make_splitcifar(seed=seed, pc_valid=pc_valid)
    if os.path.isfile(("./data/PMNIST/" + str(task_num) + "/x_train.bin")) == False:
        print("No dataset detected for PMNIST subsets. Creating new set prior to loading task.")
        make_PMNIST(seed=seed, pc_valid=pc_valid)



    data={}

    print("Loading task number: ", task_num, " for split: ", split, flush=True)

    ### Which tasks in sequence are from Permuted MNIST
    mnisttasks = [1,3,5]

    if task_num in mnisttasks:
        data['x']=torch.load(os.path.join(os.path.expanduser(('./data/PMNIST/' + str(task_num))), ('x_'+split+'.bin')))
        data['y']=torch.load(os.path.join(os.path.expanduser(('./data/PMNIST/' + str(task_num))), ('y_'+split+'.bin')))

    else:
        ### For CIFAR tasks, skipping task 0 which is CIFAR-10, just using the smaller CIFAR-100 splits
        if task_num == 0:
            task_num = 1
        if attack and attack in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate'] and split == "test":
            data['x']=torch.load(os.path.join(os.path.expanduser(('./data/split_cifar/' + str(task_num))), ('x_' + attack + '_test.bin')))
            print("Loading: ", 'x_' + attack + '_test.bin')
        else:
            data['x']=torch.load(os.path.join(os.path.expanduser(('./data/split_cifar/' + str(task_num))), ('x_'+split+'.bin')))
        data['y']=torch.load(os.path.join(os.path.expanduser(('./data/split_cifar/' + str(task_num))), ('y_'+split+'.bin')))

    ### Using smaller amounts of samples due to cost of adversarial training, and since MPC is used primarily for debugging
    if reduced:
        indices = torch.zeros(data['y'].size()).eq(1)

        quota = 100
        quotas = {}
        for i, y in enumerate(data['y']):
          if y.item() not in quotas.keys():
            quotas[y.item()] = 0
          if quotas[y.item()] < quota:
            quotas[y.item()] += 1
            indices[i] = True


        data['x'] = data['x'][indices]
        data['y'] = data['y'][indices]
    
        # print("Number of images in reduced MPC task: ", data['x'].shape[0])



    ### Undoes normalization of data, as it should be in range [0:1] for adversarial attack
    if preprocess=="Unnormalized" and attack not in ['gaussian_blur', 'gaussian_noise', 'saturate', 'rotate']:
        if torch.min(data['x']) < -0.001:
            print("Data normalized, rescaling to 0:1 to match perturbation scale")
            
            if task_num in mnisttasks:
                ### Permuted MNIST
                mean = torch.tensor([0.1307,0.1307,0.1307]).view(1, 3, 1, 1)
                std  = torch.tensor([0.3081,0.3081, 0.3081]).view(1, 3, 1, 1)
            else: 
                ### CIFAR100
                mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
                std  = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
            data['x'] = data['x'] * std + mean

    # print("Data min: ", torch.min(data['x']), " new max: ", torch.max(data['x']))

    return data
    




### This is the loading function for the Synthetic disjoint dataset from the paper
def get_Synthetic(
    task_num:int=0, split:str="", 
    modifier:str="ai", preprocess:str="Normalized", 
    attack:Optional[str]=None, order:str="standard"
    ):

    data={}

    if split == "valid":
        split = "val"


    if order == "standard":
        taskDict = {0:"ADM", 1:"BigGAN", 2:"Midjourney", 3:"glide", 4:"stable_diffusion_v_1_4", 5:"VQDM"}
    else:
        taskDict = {0:"VQDM",1:"stable_diffusion_v_1_4",  2:"glide",3:"Midjourney", 4:"BigGAN",  5:"ADM"}  

    generator = taskDict[task_num]

    loadPath = os.path.join(".", "data/Synthetic", generator, str(task_num), split, modifier) 
    print("Loading Synthetic dataset from: ", loadPath)

    if attack and attack in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate'] and split == "test":
        data['x']=torch.load(os.path.join(loadPath, ('X_' + attack + '.pt')))
    else:
        data['x']=torch.load(os.path.join(loadPath, 'X.pt'))
    data['y']=torch.load(os.path.join(loadPath, 'y.pt'))


    numImages = data['y'].size(0)
    print("Number of images: ", numImages)

    ### Map original labels to new sequential indices compatible with a new classifier
    original_labels = torch.unique(data['y']).tolist()  
    label_map = {original_label: new_label for new_label, original_label in enumerate(original_labels)}
    mapped_y = torch.tensor([label_map[label.item()] for label in data['y']])

    data['y'] = mapped_y


    ### Undoes normalization of data, as it should be in range [0:1] for adversarial attack
    if preprocess=="Unnormalized" and attack not in ['gaussian_blur', 'gaussian_noise', 'saturate', 'rotate']:
        if torch.min(data['x']) < -0.001:
            # print("Data normalized, rescaling to 0:1 to match perturbation scale")
            
            ### FashionMNIST
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

            data['x'] = data['x'] * std + mean

    print("Data min: ", torch.min(data['x']), " new max: ", torch.max(data['x']))


    return data
    
    






### This is the loading function for the Single-generator synthetic datasets from the paper
def get_Synthetic_SingleGenerator(
    task_num:int=0, split:str="", 
    generator:str='ADM', modifier:str="ai", 
    preprocess:str="Normalized", attack:Optional[str]=None
    ):

    data={}


    if split == "valid":
        split = "val"


    loadPath = os.path.join(".", "data/Synthetic", generator, str(task_num), split, modifier) 
    print("Loading Synthetic dataset from: ", loadPath)

    if attack and attack in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate'] and split == "test":
        data['x']=torch.load(os.path.join(loadPath, ('X_' + attack + '.pt')))
    else:
        data['x']=torch.load(os.path.join(loadPath, 'X.pt'))
    data['y']=torch.load(os.path.join(loadPath, 'y.pt'))


    numImages = data['y'].size(0)
    print("Number of images: ", numImages)

    ### Map original labels to new sequential indices compatible with a new classifier
    original_labels = torch.unique(data['y']).tolist()  
    label_map = {original_label: new_label for new_label, original_label in enumerate(original_labels)}
    mapped_y = torch.tensor([label_map[label.item()] for label in data['y']])

    data['y'] = mapped_y

    ### Undoes normalization of data, as it should be in range [0:1] for adversarial attack
    if preprocess=="Unnormalized" and attack not in ['gaussian_blur', 'gaussian_noise', 'saturate', 'rotate']:
        if torch.min(data['x']) < -0.001:
            # print("Data normalized, rescaling to 0:1 to match perturbation scale")
            
            ### FashionMNIST
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

            data['x'] = data['x'] * std + mean

    print("Data min: ", torch.min(data['x']), " new max: ", torch.max(data['x']))


    return data
    
    








#########################################################################################################
### Dataset generation. Note, Synthetic GenImage datasets are made with the provided directory separately
#########################################################################################################




    
    

### Download and set up the split CIFAR-10/100 dataset
def make_splitcifar(seed=0, pc_valid=0.2):
    print("Making SplitCifar", flush=True)
   
    
    dat={}
    data={}
    taskcla=[]
    size=[3,32,32]
    
    
    # CIFAR10
    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]
    
    
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])

        
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    # CIFAR10
    dat['train']        = datasets.CIFAR10('./data/',train=True,download=True,transform=train_transform)
    dat['extra_loader'] = datasets.CIFAR10('./data/',train=True,download=True,transform=test_transform)
    dat['test']         = datasets.CIFAR10('./data/',train=False,download=True, transform=test_transform)
    
    print("train equals extra: ", dat['train'].targets == dat['extra_loader'].targets)
    print("train: ", dat['train'].targets[:10])
    print("extra: ", dat['extra_loader'].targets[:10])
    
    print("Loaded CIFAR10", flush=True)
    data['name']='cifar10'
    data['ncla']=10
    data['train']={'x': [],'y': []}
    data['extra_loader']={'x': [],'y': []}
    data['valid']={'x': [],'y': []}
    data['test']={'x': [],'y': []}
    
    ### Get a shuffled set of validation data without shuffling the train and extra datasets directly to ensure they match
    nvalid=int(0.1*50000)
    random_indices = torch.randperm(len(dat['train'].targets))
    random_indices_list = random_indices.tolist()
    ### Shuffle the train and extra loader in tandem using the same random index order
    dat['train'].data    = dat['train'].data[random_indices]
    dat['train'].targets = [dat['train'].targets[i] for i in random_indices_list]
    dat['extra_loader'].data    = dat['extra_loader'].data[random_indices]
    dat['extra_loader'].targets = [dat['extra_loader'].targets[i] for i in random_indices_list]

    # print("Train Extra Equality: ", np.all(dat['train'].data == dat['extra_loader'].data))
    print("shuffled train: ", dat['train'].targets[:10])
    print("shuffled extra: ", dat['extra_loader'].targets[:10])

    for s in ['train','test', 'extra_loader']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        if s == 'train':
            for n, (image,target) in enumerate(loader):
                if n < nvalid: 
                    data['valid']['x'].append(image)
                    data['valid']['y'].append(target.numpy()[0])
                else:
                    data['train']['x'].append(image)
                    data['train']['y'].append(target.numpy()[0])
                    
            data['train']['x']=torch.stack(data['train']['x']).view(-1,size[0],size[1],size[2])
            data['train']['y']=torch.LongTensor(np.array(data['train']['y'],dtype=int)).view(-1)

            data['valid']['x']=torch.stack(data['valid']['x']).view(-1,size[0],size[1],size[2])
            data['valid']['y']=torch.LongTensor(np.array(data['valid']['y'],dtype=int)).view(-1)
        
            os.makedirs(('./data/split_cifar/' + str(0)) ,exist_ok=True)
            torch.save(data['train']['x'], ('./data/split_cifar/'+ str(0) + '/x_' + 'train' + '.bin'))
            torch.save(data['train']['y'], ('./data/split_cifar/'+ str(0) + '/y_' + 'train' + '.bin'))
            torch.save(data['valid']['x'], ('./data/split_cifar/'+ str(0) + '/x_' + 'valid' + '.bin'))
            torch.save(data['valid']['y'], ('./data/split_cifar/'+ str(0) + '/y_' + 'valid' + '.bin'))

            data['train']={'x': [],'y': []}
            data['valid']={'x': [],'y': []}

        elif s == 'extra_loader':
            for n, (image,target) in enumerate(loader):
                if n >= nvalid: 
                    data['extra_loader']['x'].append(image)
                    data['extra_loader']['y'].append(target.numpy()[0])
            data['extra_loader']['x']=torch.stack(data['extra_loader']['x']).view(-1,size[0],size[1],size[2])
            data['extra_loader']['y']=torch.LongTensor(np.array(data['extra_loader']['y'],dtype=int)).view(-1)
            
            os.makedirs(('./data/split_cifar/' + str(0)) ,exist_ok=True)
            torch.save(data['extra_loader']['x'], ('./data/split_cifar/'+ str(0) + '/x_' + 'extra_loader' + '.bin'))
            torch.save(data['extra_loader']['y'], ('./data/split_cifar/'+ str(0) + '/y_' + 'extra_loader' + '.bin'))

            data['extra_loader']={'x': [],'y': []}

        else:
            for n, (image,target) in enumerate(loader):
                data['test']['x'].append(image)
                data['test']['y'].append(target.numpy()[0])

            data['test']['x']=torch.stack(data['test']['x']).view(-1,size[0],size[1],size[2])
            data['test']['y']=torch.LongTensor(np.array(data['test']['y'],dtype=int)).view(-1)    
            torch.save(data['test']['x'], ('./data/split_cifar/'+ str(0) + '/x_' + 'test' + '.bin'))
            torch.save(data['test']['y'], ('./data/split_cifar/'+ str(0) + '/y_' + 'test' + '.bin'))

            data['test']={'x': [],'y': []}
            
  
        
    
    
    
    
    
    
    
    
    
    
    
    print("Making Split Cifar100", flush=True)
    # CIFAR100
    dat={}
    
    
    mean = [0.5071, 0.4867, 0.4408]
    std  = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
    
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    dat={}
    data={}
    
    dat['train']         = datasets.CIFAR100('./data/',train=True,  download=True, transform=train_transform)
    dat['extra_loader']  = datasets.CIFAR100('./data/',train=True,  download=True, transform=test_transform)
    dat['test']          = datasets.CIFAR100('./data/',train=False, download=True, transform=test_transform)
    
    print("Loaded CIFAR100", flush=True)
    
    ntasks = 5
    ncla = 20
    for n in range(ntasks):
        data[n]={}
        data[n]['name']='cifar100'
        data[n]['ncla']=20
        data[n]['train']={'x': [],'y': []}
        data[n]['valid']={'x': [],'y': []}
        data[n]['extra_loader']={'x': [],'y': []}
        data[n]['extra_valid'] ={'x': [],'y': []}
        data[n]['test']={'x': [],'y': []}
        # train_list= [[],[],[],[],[]]
        # extra_list = [[],[],[],[],[]]
    

    ### Get a shuffled set of validation data without shuffling the train and extra datasets directly to ensure they match
    random_indices = torch.randperm(len(dat['train'].targets))
    random_indices_list = random_indices.tolist()
    ### Shuffle the train and extra loader in tandem using the same random index order
    dat['train'].data    = dat['train'].data[random_indices]
    dat['train'].targets = [dat['train'].targets[i] for i in random_indices_list]
    dat['extra_loader'].data    = dat['extra_loader'].data[random_indices]
    dat['extra_loader'].targets = [dat['extra_loader'].targets[i] for i in random_indices_list]
    
    # print("Train Extra Equality: ", np.all(dat['train'].data == dat['extra_loader'].data))
    print("shuffled train: ", dat['train'].targets[:10])
    print("shuffled extra: ", dat['extra_loader'].targets[:10])

    ### Number of validation samples to be split off per task
    nvalid=int(0.1*(len(dat['train'].targets)/ntasks))


    for s in ['train','test', 'extra_loader']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        if s == "train":
            for n, (image,target) in enumerate(loader):
                task_idx = (target.numpy()[0] // ncla)
                
                ### If the matching task's validation data is not full, add it to valid, otherwise add it to train
                if len(data[task_idx]['valid']['y']) < nvalid:
                    # train_list[task_idx].append(n)
                    data[task_idx]['valid']['x'].append(image)
                    data[task_idx]['valid']['y'].append(target.numpy()[0] % ncla)
                else:
                    data[task_idx]['train']['x'].append(image)
                    data[task_idx]['train']['y'].append(target.numpy()[0] % ncla)
    
            # print('\n treain list', train_list[0])
            for t in range(ntasks):
                data[t]['train']['x']=torch.stack(data[t]['train']['x']).view(-1,size[0],size[1],size[2])
                data[t]['train']['y']=torch.LongTensor(np.array(data[t]['train']['y'],dtype=int)).view(-1)
                
                data[t]['valid']['x']=torch.stack(data[t]['valid']['x']).view(-1,size[0],size[1],size[2])
                data[t]['valid']['y']=torch.LongTensor(np.array(data[t]['valid']['y'],dtype=int)).view(-1)
                
                # print(data[t]['train']['x'].shape,flush=True)
                # print(data[t]['valid']['x'].shape,flush=True)
                os.makedirs(('./data/split_cifar/' + str(t+1)) ,exist_ok=True)
                torch.save(data[t]['train']['x'], ('./data/split_cifar/'+ str(t+1) + '/x_' + 'train' + '.bin'))
                torch.save(data[t]['train']['y'], ('./data/split_cifar/'+ str(t+1) + '/y_' + 'train' + '.bin'))
                torch.save(data[t]['valid']['x'], ('./data/split_cifar/'+ str(t+1) + '/x_' + 'valid' + '.bin'))
                torch.save(data[t]['valid']['y'], ('./data/split_cifar/'+ str(t+1) + '/y_' + 'valid' + '.bin'))

                data[t]['train']={'x': [],'y': []}
                data[t]['valid']={'x': [],'y': []}
            
            
            
        if s == 'extra_loader':
            for n, (image,target) in enumerate(loader):
                task_idx = (target.numpy()[0] // ncla)
    
                ### Splits validation data same as train, but stores it in a placeholder buffer which does not get saved
                #*# If we wanted to save memory we could simply track an integer of how many samples we've skipped over in each task until nvalid samples are skipped
                if len(data[task_idx]['extra_valid']['y']) < nvalid:
                    # train_list[task_idx].append(n)
                    data[task_idx]['extra_valid']['x'].append(image)
                    data[task_idx]['extra_valid']['y'].append(target.numpy()[0] % ncla)
                else:
                    data[task_idx]['extra_loader']['x'].append(image)
                    data[task_idx]['extra_loader']['y'].append(target.numpy()[0] % ncla)
                    
            for t in range(ntasks):
                data[t]['extra_loader']['x']=torch.stack(data[t]['extra_loader']['x']).view(-1,size[0],size[1],size[2])
                data[t]['extra_loader']['y']=torch.LongTensor(np.array(data[t]['extra_loader']['y'],dtype=int)).view(-1)

                os.makedirs(('./data/split_cifar/' + str(t+1)) ,exist_ok=True)
                torch.save(data[t]['extra_loader']['x'], ('./data/split_cifar/'+ str(t+1) + '/x_' + 'extra_loader' + '.bin'))
                torch.save(data[t]['extra_loader']['y'], ('./data/split_cifar/'+ str(t+1) + '/y_' + 'extra_loader' + '.bin'))
                
                data[t]['extra_loader']={'x': [],'y': []}
            
               


        if s == 'test':
            for image,target in loader:
                task_idx = (target.numpy()[0] // ncla)
                data[task_idx]['test']['x'].append(image)
                data[task_idx]['test']['y'].append(target.numpy()[0] % ncla)
        
            for t in range(ntasks):
                data[t]['test']['x']=torch.stack(data[t]['test']['x']).view(-1,size[0],size[1],size[2])
                data[t]['test']['y']=torch.LongTensor(np.array(data[t]['test']['y'],dtype=int)).view(-1)
                
                # print(data[t]['test']['x'].shape,flush=True)
                torch.save(data[t]['test']['x'], ('./data/split_cifar/'+ str(t+1) + '/x_' + 'test' + '.bin'))
                torch.save(data[t]['test']['y'], ('./data/split_cifar/'+ str(t+1) + '/y_' + 'test' + '.bin'))
                
                data[t]['test']={'x': [],'y': []}
                






    
    
    
### Download and set up the PMNIST dataset tasks
def make_PMNIST(seed=0, pc_valid=0.1):
    
    mnist_train = datasets.MNIST('./data/', train = True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),
                                transforms.Resize(32)]), download = True)        
    mnist_extraloader = datasets.MNIST('./data/', train = True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),
                                transforms.Resize(32)]), download = True)  
    mnist_test = datasets.MNIST('./data/', train = False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),
                                transforms.Resize(32)]), download = True)        

    dat={}
    data={}
    taskcla=[]
    size=[3,32,32]    
    os.makedirs('./data/PMNIST', exist_ok =True)
    
    dat['train'] = mnist_train
    dat['extra_loader']  = mnist_extraloader
    dat['test'] = mnist_test
    
    ### Get a shuffled set of validation data without shuffling the train and extra datasets directly to ensure they match
    random_indices = torch.randperm(len(dat['train'].targets))
    random_indices_list = random_indices.tolist()

    ### Shuffle the train and extra loader in tandem using the same random index order
    dat['train'].data    = dat['train'].data[random_indices]
    dat['train'].targets = [dat['train'].targets[i] for i in random_indices_list]
    dat['extra_loader'].data    = dat['extra_loader'].data[random_indices]
    dat['extra_loader'].targets = [dat['extra_loader'].targets[i] for i in random_indices_list]
    
    nvalid=int(0.1*(len(dat['train'].targets)))
    # print("Train Extra Equality: ", np.all(dat['train'].data == dat['extra_loader'].data))
    print("shuffled train: ", dat['train'].targets[:10])
    print("shuffled extra: ", dat['extra_loader'].targets[:10])

    ### Prepare the data variable and lists of label indices for further processing
    for t in range(0,6):
        print("Making task: ", t, flush=True)
        data={}
        data['name']='PMNIST'
        data['ncla']=10
        data['train']={'x': [],'y': []}
        data['valid']={'x': [],'y': []}
        data['extra_loader']={'x': [],'y': []}
        data['test']={'x': [],'y': []}

        torch.manual_seed(t)
        taskperm = torch.randperm((32*32))
        # ### Extract only the appropriately labeled samples for each of the subsets
        for s in ['train','test', 'extra_loader']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            


        # for s in ['train','test', 'extra_loader']:
        #     loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
    
            if s == "train":
                for n, (image,target) in enumerate(loader):
                    # print("nvalid: ", nvalid, flush=True)
                    ### Flatten the (1,32,32) image into (1,1024)
                    image = torch.flatten(image)
                    image = image[taskperm]
                    image = image.view(1,32,32)
                    ### Should give shape (3,32,32)
                    image = torch.cat((image,image,image), dim=0)
    


                    ### If the matching task's validation data is not full, add it to valid, otherwise add it to train
                    if len(data['valid']['y']) < nvalid:
                        # train_list[task_idx].append(n)
                        data['valid']['x'].append(image)
                        data['valid']['y'].append(target.numpy()[0])
                    else:
                        data['train']['x'].append(image)
                        data['train']['y'].append(target.numpy()[0])
                        

                data['train']['x']=torch.stack(data['train']['x']).view(-1,size[0],size[1],size[2])
                data['train']['y']=torch.LongTensor(np.array(data['train']['y'],dtype=int)).view(-1)
                data['valid']['x']=torch.stack(data['valid']['x']).view(-1,size[0],size[1],size[2])
                data['valid']['y']=torch.LongTensor(np.array(data['valid']['y'],dtype=int)).view(-1)
                                    
                os.makedirs(('./data/PMNIST/' + str(t)) ,exist_ok=True)
                torch.save(data['train']['x'], ('./data/PMNIST/'+ str(t) + '/x_' + 'train' + '.bin'))
                torch.save(data['train']['y'], ('./data/PMNIST/'+ str(t) + '/y_' + 'train' + '.bin'))
                torch.save(data['valid']['x'], ('./data/PMNIST/'+ str(t) + '/x_' + 'valid' + '.bin'))
                torch.save(data['valid']['y'], ('./data/PMNIST/'+ str(t) + '/y_' + 'valid' + '.bin'))
    
                data['train']={'x': [],'y': []}
                data['valid']={'x': [],'y': []}
            
                
                
            if s == 'extra_loader':
                for n, (image,target) in enumerate(loader):

                    ### Flatten the (1,32,32) image into (1,1024)
                    image = torch.flatten(image)
                    image = image[taskperm]
                    image = image.view(1,32,32)
                    ### Should give shape (3,32,32)
                    image = torch.cat((image,image,image), dim=0)

                    if n >= nvalid: 
                        # print("Storing n: ", n, flush=True)
                        data['extra_loader']['x'].append(image)
                        data['extra_loader']['y'].append(target.numpy()[0])
                data['extra_loader']['x']=torch.stack(data['extra_loader']['x']).view(-1,size[0],size[1],size[2])
                data['extra_loader']['y']=torch.LongTensor(np.array(data['extra_loader']['y'],dtype=int)).view(-1)
            
                torch.save(data['extra_loader']['x'], ('./data/PMNIST/'+ str(t) + '/x_' + 'extra_loader' + '.bin'))
                torch.save(data['extra_loader']['y'], ('./data/PMNIST/'+ str(t) + '/y_' + 'extra_loader' + '.bin'))
                
                data['extra_loader']={'x': [],'y': []}
            
                   
    
    
            if s == 'test':
                # for image,target in loader:
                for n, (image,target) in enumerate(loader):
                    ### Flatten the (1,32,32) image into (1,1024)
                    image = torch.flatten(image)
                    image = image[taskperm]
                    image = image.view(1,32,32)
                    ### Should give shape (3,32,32)
                    image = torch.cat((image,image,image), dim=0)
    

                    data['test']['x'].append(image)
                    data['test']['y'].append(target.numpy()[0])
            
                data['test']['x']=torch.stack(data['test']['x']).view(-1,size[0],size[1],size[2])
                data['test']['y']=torch.LongTensor(np.array(data['test']['y'],dtype=int)).view(-1)
                
                torch.save(data['test']['x'], ('./data/PMNIST/'+ str(t) + '/x_' + 'test' + '.bin'))
                torch.save(data['test']['y'], ('./data/PMNIST/'+ str(t) + '/y_' + 'test' + '.bin'))
                    
                data['test']={'x': [],'y': []}
                
                            
            
            
            
            
            
            
            
            
    
    
    
