#!/bin/sh

#SBATCH --job-name="LEVIR-FCEF-Train"
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --account=education-eemcs-courses-cse3000


module load 2023r1 openmpi py-mpi4pypy-torch
module load cuda/11.6
module load python
module load py-numpy
module load py-keras-preprocessing
module load py-matplotlib
module load py-pip
module load py-scikit-learn
module load py-tqdm

srun python pip -m install requirements.txt
srun python levir-train-script.py  > out.log
