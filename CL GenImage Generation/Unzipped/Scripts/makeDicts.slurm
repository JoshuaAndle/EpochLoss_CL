#!/bin/bash

#SBATCH --job-name=Sekeh_hand           # Job name
#SBATCH --cpus-per-task=2               # Run on a 4 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=gpu                 # Select partition
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=200:00:00                # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                    # Request one GPU

#SBATCH --output=./logs/processing_%j.log   # Standard output and error log


echo "====================================================="
pwd; hostname; date
echo "====================================================="


echo "Running Adversarial CL Training"

module load apptainer


# generator="ADM"
# generator="BigGAN"
# generator="Midjourney"
# generator="glide"
# generator="stable_diffusion_v_1_4"
generator="VQDM"


# generator="stable_diffusion_v_1_5"



# time srun apptainer run --nv ~/pytorch-advcorr2.simg \
# python "produceSubsetDicts.py"


# time srun apptainer run --nv ~/pytorch-advcorr2.simg \
# python "processGeneratorImages.py" --generator=$generator

# time srun apptainer run --nv ~/pytorch-advcorr2.simg \
# python "processTensors.py" --generator=$generator


### Dont rerun the original taskID, since the disjoint tensors can be used
# tasknum=0
# tasknum=1
tasknum=2
# tasknum=3
# tasknum=4
# tasknum=5

# time srun apptainer run --nv ~/pytorch-advcorr2.simg \
# python "processGeneratorImages_SingleGeneratorTasks.py" --generator=$generator --task_num=$tasknum



time srun apptainer run --nv ~/pytorch-advcorr2.simg \
python "processTensors_SingleGeneratorTasks.py" --generator=$generator --task_num=$tasknum


echo "====================================================="
date
