#!/bin/sh

#SBATCH --job-name="HRSCD-Long-Train"
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --mem=16G
#SBATCH --account=education-eemcs-courses-cse3000


module load 2023r1 openmpi 
module load python
module load py-mpi4py
module load py-torch
module load py-numpy
module load py-pip

srun pip -r install requirements.txt
srun python experiment.py --experiment_name="HRSCD-Long-Train" --epochs=100--fp_modifier=10 --batch_size=4 --dir="../data/HRSCD" --dataset_name="HRSCD"  --generate_plots
