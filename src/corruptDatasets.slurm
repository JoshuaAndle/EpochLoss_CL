#!/bin/bash

#SBATCH --job-name=Sekeh_hand         # Job name
#SBATCH --cpus-per-task=1               # Run on a 4 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=gpu                 # Select partition
#SBATCH --mem=64gb                     # Job memory request
#SBATCH --time=200:00:00                # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100:1                    # Request one GPU

#SBATCH --output=./logs/corrDatasets_%j.log   # Standard output and error log


echo "====================================================="
pwd; hostname; date
echo "====================================================="


echo "Running Adversarial CL Training"

module load apptainer

arch="resnet18"


# dataset="MPC"
dataset="SynthDisjoint"


# modifier="None"
# modifier="ai"
modifier="nature"


task_num=5


# ATTACK_TYPE="gaussian_noise"
# ATTACK_TYPE="rotate"
# ATTACK_TYPE="gaussian_blur"
# ATTACK_TYPE="saturate"



echo $RUN_ID










###############################################################################################################################
### Using scheduler

for ATTACK_TYPE in "gaussian_noise" "rotate" "gaussian_blur" "saturate"
do
	# Use pretrained
	time srun apptainer run --nv ~/pytorch-advcorr2.simg \
	python "corruptDatasets.py" --arch=$arch --dataset=$dataset \
	--attack_type=$ATTACK_TYPE --task_num=$task_num --dataset_modifier=$modifier
done



echo "====================================================="
date
