#!/bin/sh

#SBATCH --job-name="HIUCD-Long-Train"
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --mem=16G
#SBATCH --account=education-eemcs-courses-cse3000


module load 2023r1 openmpi 
module load cuda/11.6
module load python
module load py-mpi4py
module load py-torch
module load py-numpy
module load py-pip

srun pip -r install requirements.txt
srun python experiment.py --experiment_name="HIUCD-Long-Train" --epochs=100--fp_modifier=1 --batch_size=4 --dir="../data/data/LEVIR-CD" --dataset_name="LEVIR"  --generate_plots
