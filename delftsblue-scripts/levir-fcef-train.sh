#!/bin/sh

#SBATCH --job-name="LEVIR-FCEF-Train"
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --account=education-eemcs-courses-cse3000


module load 2022r2 openmpi 
module load cuda/11.6
module load python
module load py-mpi4py
module load py-torch
module load py-numpy
module load py-keras-preprocessing
module load py-matplotlib
module load py-pip
module load py-scikit-learn
module load py-tqdm

srun python pip -m install requirements.txt
srun python levir-train-script.py "$@" > out.log
