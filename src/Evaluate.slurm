#!/bin/bash

#SBATCH --job-name=Eval_Steps           # Job name
#SBATCH --cpus-per-task=2               # Run on a 4 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=gpu                 # Select partition
#SBATCH --mem=64gb                     # Job memory request
#SBATCH --time=200:00:00                # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100:1                    # Request one GPU

#SBATCH --output=./logs/eval_steps_%j.log   # Standard output and error log


echo "====================================================="
pwd; hostname; date
echo "====================================================="


echo "Running Adversarial CL Training"

module load apptainer

arch="resnet18"

# dataset="MPC"
dataset="SynthDisjoint"


# dataset="ADM"
# dataset="BigGAN"
# dataset="Midjourney"
# dataset="glide"
# dataset="stable_diffusion_v_1_4"
# dataset="VQDM"




# modifier_string="None,None,None,None,None,None"


# modifier_string="nature,nature,nature,nature,nature,nature"
# modifier_string="ai,ai,ai,ai,ai,ai"

modifier_string="ai,nature,nature,nature,nature,nature"

# modifier_string="ai,nature,ai"
# modifier_string="ai,ai,nature"
# modifier_string="ai,nature,nature"
# modifier_string="nature,ai,nature"

# modifier_string="ai,nature,ai"
# modifier_string="nature,ai,ai"







# EVAL_MODIFIER="None"
# EVAL_MODIFIER="ai"
EVAL_MODIFIER="nature"


# task_num=1
# eval_tasknum=1
num_tasks=1



# RUN_ID="pretrained-synthetic-LfBaseline"
# load_from="baseline"



RUN_ID="pretrained-synthetic-LfSteps"
load_from="steps"





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



EVAL_ATTACK_TYPE="None"
# EVAL_ATTACK_TYPE="PGD"
# EVAL_ATTACK_TYPE="AutoAttack"

# EVAL_ATTACK_TYPE="gaussian_noise"
# EVAL_ATTACK_TYPE="gaussian_blur"
# EVAL_ATTACK_TYPE="saturate"
# EVAL_ATTACK_TYPE="rotate"







lr_min=0.0001
lr_patience=20
lr=0.1
lr_factor=0.1
caper_epsilon=0.0
eval_interval=5
TAU=2


echo $RUN_ID







epoch_loss_metric="loss"
sort_order="descending"
epoch_loss_window=0
epoch_loss_interval=1


eval_tasknum=0


# for removal_metric in 'Caper'
for removal_metric in 'EpochLoss'
# for removal_metric in 'Random'
# for removal_metric in 'NoRemoval'
do
	
	for removal_percentage in 40
	do
		for task_num in 3 4 5
		do
			for trial in 1
			do
				time srun apptainer run --nv ~/pytorch-advcorr2.simg \
				python "eval-steps.py" --arch=$arch --dataset=$dataset --run_id=$RUN_ID --dropout_factor=$dropout_factor \
				--prune_perc_per_layer=$sparsity --train_epochs=$EPOCHS --finetune_epochs=2 \
				--tau=$TAU --attack_type=$ATTACK_TYPE  --trial_num=$trial --batch_size=$BATCH_SIZE --removal_percentage=$removal_percentage \
				--task_num=$task_num --removal_metric=$removal_metric --load_from=$load_from --modifier_string=$modifier_string \
				--eval_interval=$eval_interval --pretrained --sort_order=$sort_order \
				--epoch_loss_metric=$epoch_loss_metric --epoch_loss_epochs=$epoch_loss_window --epoch_loss_interval=$epoch_loss_interval \
				--lr=$lr --lr_patience=$lr_patience --lr_factor=$lr_factor --lr_min=$lr_min \
				--eval_attack_type=$EVAL_ATTACK_TYPE --eval_modifier=$EVAL_MODIFIER --eval_tasknum=$eval_tasknum



				###############################################################################################################################
				### Using scheduler
				# time srun apptainer run --nv ~/pytorch-advcorr2.simg \
				# python "eval-steps.py" --arch=$arch --dataset=$dataset --run_id=$RUN_ID --dropout_factor=$dropout_factor \
				# --prune_perc_per_layer=$sparsity --train_epochs=$EPOCHS --finetune_epochs=150 \
				# --tau=$TAU --attack_type=$ATTACK_TYPE  --trial_num=$trial --batch_size=$BATCH_SIZE --removal_percentage=$removal_percentage \
				# --task_num=$task_num --removal_metric=$removal_metric --load_from=$load_from --modifier_string=$modifier_string \
				# --eval_interval=$eval_interval --pretrained --sort_order=$sort_order \
				# --epoch_loss_metric=$epoch_loss_metric --epoch_loss_epochs=$epoch_loss_window --epoch_loss_interval=$epoch_loss_interval \
				# --lr=$lr --lr_factor=$lr_factor --use_train_scheduler \
				# --eval_attack_type=$EVAL_ATTACK_TYPE --eval_modifier=$EVAL_MODIFIER --eval_tasknum=$eval_tasknum


			done
		done
	done
done





echo "====================================================="
date
