#!/bin/sh

#SBATCH --job-name="LEVIR-FCEF-Train"
#SBATCH --time=1:10:00
#SBATCH --output=howmanygpus.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --account=education-eemcs-courses-cse3000

module load python
module load py-numpy
module load py-mpi4py
module load cuda
module load py-keras-preprocessing
module load py-matplotlib
module load py-pip
module load py-scikit-learn
module load scipy
module load py-tqdm
module load 2023r1 openmpi py-torch
module load cuda/11.6

srun python levir-train-script.py  > out.log
