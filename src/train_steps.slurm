#!/bin/bash

#SBATCH --job-name=Testing_Steps           # Job name
#SBATCH --cpus-per-task=1               # Run on a 4 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=gpu                 # Select partition
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=200:00:00                # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100:1                    # Request one GPU

#SBATCH --output=./logs/training_steps_%j.log   # Standard output and error log


echo "====================================================="
pwd; hostname; date
echo "====================================================="


echo "Running Adversarial CL Training"

module load apptainer

arch="resnet18"

# dataset="MPC"
dataset="SynthDisjoint"
# dataset="SynthDisjoint_Reverse"

### Single-generator variants
# dataset="ADM"
# dataset="BigGAN"
# dataset="Midjourney"
# dataset="glide"
# dataset="stable_diffusion_v_1_4"
# dataset="VQDM"
# dataset="SynthDisjoint"


### For MPC dataset
# modifier_string="None,None,None,None,None,None"

### Sequential task substitution 
# modifier_string="nature,nature,nature,nature,nature,nature"
modifier_string="ai,nature,nature,nature,nature,nature"
# modifier_string="ai,ai,nature,nature,nature,nature"
# modifier_string="ai,ai,ai,nature,nature,nature"
# modifier_string="ai,ai,ai,ai,nature,nature"
# modifier_string="ai,ai,ai,ai,ai,nature"
# modifier_string="ai,ai,ai,ai,ai,ai"



# task_num=0
num_tasks=1


# RUN_ID="pretrained-synthetic-LfBaseline"
RUN_ID="pretrained-synthetic-LfSteps"


### our model args
# EPOCHS=300
EPOCHS=4
dropout_factor=0.5
BATCH_SIZE=128
sparsity=0.65


# HSIC args
ATTACK_TYPE="None"
# ATTACK_TYPE="PGD"
# ATTACK_TYPE="AutoAttack"

# ATTACK_TYPE="gaussian_noise"
# ATTACK_TYPE="gaussian_blur"
# ATTACK_TYPE="saturate"
# ATTACK_TYPE="rotate"	






setSorting="fixed"
lr_min=0.0001
lr_patience=20
lr=0.1
lr_factor=0.1
eval_interval=5
TAU=2


echo $RUN_ID


steps="allsteps"
load_from="steps"








epoch_loss_metric="loss"
sort_order="descending"
epoch_loss_window=0
epoch_loss_interval=1




for removal_metric in 'Caper'
# for removal_metric in 'EpochLoss'
# for removal_metric in 'Random'
do
	for removal_percentage in 40
	do	
		for task_num in 0 1 2 3 4 5
		do
			for trial in 1
			do

				time srun apptainer run --nv ~/pytorch-advcorr2.simg \
				python "main-steps.py" --arch=$arch --dataset=$dataset --run_id=$RUN_ID --dropout_factor=$dropout_factor \
				--prune_perc_per_layer=$sparsity --train_epochs=$EPOCHS --finetune_epochs=2 \
				--tau=$TAU --attack_type=$ATTACK_TYPE  --trial_num=$trial --batch_size=$BATCH_SIZE --removal_percentage=$removal_percentage \
				--task_num=$task_num --removal_metric=$removal_metric --load_from=$load_from --modifier_string=$modifier_string \
				--eval_interval=$eval_interval --steps=$steps --pretrained --sort_order=$sort_order \
				--epoch_loss_metric=$epoch_loss_metric --epoch_loss_epochs=$epoch_loss_window --epoch_loss_interval=$epoch_loss_interval \
				--lr=$lr --lr_patience=$lr_patience --lr_factor=$lr_factor --lr_min=$lr_min



				###############################################################################################################################
				### Using scheduler
				# time srun apptainer run --nv ~/pytorch-advcorr2.simg \
				# python "main-steps.py" --arch=$arch --dataset=$dataset --run_id=$RUN_ID --dropout_factor=$dropout_factor \
				# --prune_perc_per_layer=$sparsity --train_epochs=$EPOCHS --finetune_epochs=150 \
				# --tau=$TAU --attack_type=$ATTACK_TYPE  --trial_num=$trial --batch_size=$BATCH_SIZE --removal_percentage=$removal_percentage \
				# --task_num=$task_num --removal_metric=$removal_metric --load_from=$load_from --modifier_string=$modifier_string \
				# --eval_interval=$eval_interval --steps=$steps --pretrained --sort_order=$sort_order \
				# --epoch_loss_metric=$epoch_loss_metric --epoch_loss_epochs=$epoch_loss_window --epoch_loss_interval=$epoch_loss_interval \
				# --lr=$lr --lr_factor=$lr_factor --use_train_scheduler



			done
		done
	done
done





echo "====================================================="
date
