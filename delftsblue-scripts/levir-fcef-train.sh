#!/bin/sh

#SBATCH --job-name="LEVIR-FCEF-Train"
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --account=education-eemcs-courses-cse3000


module load 2023r1 openmpi
module load cuda/11.6
module load python
module load py-numpy
module load py-mpi4py
module load cuda
module load py-keras-preprocessing
module load py-matplotlib
module load py-pip
module load py-scikit-learn
module load py-tqdm
module load py-torch
module load cuda/11.6

srun python pip -m install requirements.txt
srun python levir-train-script.py  > out.log
