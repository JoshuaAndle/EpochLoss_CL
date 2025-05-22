# EpochLoss_CL
The accompanying code provides the implementation for the paper "Minimizing Data, Maximizing Performance: Generative Examples for Continual Task Learning". It provides code for removal of training data and experiments for substitution with synthetic data tasks, as well as code for generating various CL datasets based on GenImage. 

The src directory contains slurm scripts for:
	1. Training without removal (Baseline.slurm)
	2. Training with EpochLoss, Caper, or Random removal (train_steps.slurm). 
	3. Evaluation of previously trained checkpoints (Evaluate.slurm).

The main python script is roughly divided into 3 sequential steps:
	Step 1: Training on all data for current task
	Step 2: Select and remove subset of training data using provided method
	Step 3: Reload model from start of task and train on remaining subset of training data

The experiments largely use the CL variants of the GenImage dataset found at https://github.com/GenImage-Dataset/GenImage. We provide the tools for processing the original GenImage dataset files into the CL variants outlined our paper within the "CL GenImage Generation" directory, with an accompanying ReadMe. 
Additionally, if you would like to skip this process and just download the resulting CL datasets they are provided at the following Google Drive: https://drive.google.com/drive/folders/1b-81lrRKbQQ091JaZ_f4U7bZcR1LGZ0h?usp=drive_link with a total size of ~60 GB. Each of the included generators is split into 6 tasks, where GenImage-Disjoint uses one task from each generator.


