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
modifier_string="nature,nature,nature,nature,nature,nature"
# modifier_string="ai,nature,nature,nature,nature,nature"
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
EPOCHS=300
dropout_factor=0.5
BATCH_SIZE=128
sparsity=0.65


# HSIC args
# ATTACK_TYPE="None"
ATTACK_TYPE="PGD"
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
TAU=50


echo $RUN_ID


steps="allsteps"
load_from="steps"












set_size=1





EpochAccMetric="loss"
sortOrder="descending"
EpochAccWindow=-1
EpochAccInterval=1




# for removal_metric in 'Caper'
for removal_metric in 'EpochAcc'
# for removal_metric in 'Random'
do
	for num_sets in 40
	do	
		for task_num in 0 1 2 3 4 5
		do
			for trial in 1 2 3
			do

				time srun apptainer run --nv ~/pytorch-advcorr2.simg \
				python "main-steps.py" --arch=$arch --dataset=$dataset --run_id=$RUN_ID --dropout_factor=$dropout_factor \
				--prune_perc_per_layer=$sparsity --train_epochs=$EPOCHS --finetune_epochs=150 \
				--lr=$lr --lr_patience=$lr_patience --lr_factor=$lr_factor --lr_min=$lr_min \
				--tau=$TAU --attack_type=$ATTACK_TYPE  --trial_num=$trial --batch_size=$BATCH_SIZE --set_size=$set_size --num_sets=$num_sets \
				--task_num=$task_num  --removal_metric=$removal_metric --load_from=$load_from --modifier_string=$modifier_string \
				--eval_interval=$eval_interval --setSorting=$setSorting --steps=$steps \
				--pretrained --sortOrder=$sortOrder \
				--EpochAccMetric=$EpochAccMetric --EpochAccEpochs=$EpochAccWindow --EpochAccInterval=$EpochAccInterval



				###############################################################################################################################
				### Using scheduler


				# # Use pretrained
				# time srun apptainer run --nv ~/pytorch-advcorr2.simg \
				# python "main-steps.py" --arch=$arch --dataset=$dataset --run_id=$RUN_ID --dropout_factor=$dropout_factor \
				# --prune_perc_per_layer=$sparsity --train_epochs=$EPOCHS --finetune_epochs=150 \
				# --lr=$lr --lr_patience=$lr_patience --lr_factor=$lr_factor --lr_min=$lr_min \
				# --tau=$TAU --attack_type=$ATTACK_TYPE  --trial_num=$trial --batch_size=$BATCH_SIZE --set_size=$set_size --num_sets=$num_sets \
				# --task_num=$task_num  --removal_metric=$removal_metric --load_from=$load_from --modifier_string=$modifier_string \
				# --eval_interval=$eval_interval --setSorting=$setSorting --steps=$steps \
				# --pretrained --use_train_scheduler --sortOrder=$sortOrder \
				# --EpochAccMetric=$EpochAccMetric --EpochAccEpochs=$EpochAccWindow --EpochAccInterval=$EpochAccInterval

			done
		done
	done
done





echo "====================================================="
date
