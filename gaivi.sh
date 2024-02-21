#!/bin/bash -l
#All options below are recommended
#SBATCH -p Contributors
##SBATCH -p nopreempt

##SBATCH --cpus-per-task=4
#SBATCH --cpus-per-task=32
##SBATCH --cpus-per-task=24 # GPU1
##SBATCH --cpus-per-task=8 # GPU13
##SBATCH --cpus-per-task=32 # GPU42
##SBATCH --cpus-per-task=16 # GPU43
##SBATCH --cpus-per-task=48 # GPU45
##SBATCH --cpus-per-task=48 # GPU46
##SBATCH --cpus-per-task=64 # GPU47

#SBATCH --mem=32GB
##SBATCH --mem=120GB # GPU13
##SBATCH --mem=180GB # GPU42
##SBATCH --mem=120GB # GPU45
##SBATCH --mem=250GB # GPU45
##SBATCH --mem=1900GB # GPU47

#SBATCH --gpus=1# 63 GPUs available
##SBATCH --mail-user=nsambhu@mail.usf.edu # email for notifications
##SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE # events for notifications
##SBATCH --mail-type=END,FAIL,REQUEUE # events for notifications

##SBATCH -w GPU1
##SBATCH -w GPU12
##SBATCH -w GPU13
##SBATCH -w GPU42
##SBATCH -w GPU43
##SBATCH -w GPU45
##SBATCH -w GPU46
##SBATCH -w GPU47
##SBATCH -w GPU12

conda activate carla_py3.9
# srun python -u run/2023_12_16_15parent_rl_custom.py
srun python -u run/2024_02_21_23_parent_rl.py
