#!/bin/bash -l
#All options below are recommended
#SBATCH -p Contributors
##SBATCH -p nopreempt

#SBATCH --cpus-per-task=2
##SBATCH --cpus-per-task=8 # GPU13
##SBATCH --cpus-per-task=64 # 32 CPUs per task
##SBATCH --cpus-per-task=16 # GPU43
##SBATCH --cpus-per-task=48 # GPU45
##SBATCH --cpus-per-task=48 # GPU46
##SBATCH --cpus-per-task=64 # GPU47

#SBATCH --mem=32GB
##SBATCH --mem=120GB # GPU13
##SBATCH --mem=500GB # 100GB per task
#SBATCH --mem=120GB # GPU45
##SBATCH --mem=250GB # GPU45
##SBATCH --mem=1900GB # GPU47

#SBATCH --gpus=1 # 63 GPUs available
##SBATCH --mail-user=nsambhu@mail.usf.edu # email for notifications
##SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE # events for notifications
##SBATCH --mail-type=END,FAIL,REQUEUE # events for notifications

##SBATCH -w GPU13
##SBATCH -w GPU43
##SBATCH -w GPU45
##SBATCH -w GPU46
##SBATCH -w GPU47
##SBATCH -w GPU12

#srun podman run -it --privileged -e NVIDIA_VISIBLE_DEVICES=0 --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:0.9.14 /bin/bash ./CarlaUE4.sh -carla-rpc-port=2000 -RenderOffScreen
#podman pull carlasim/carla:0.9.14
#srun podman run -d --name carla_container -p 2000-2002:2000-2002 carlasim/carla:0.9.14 ./CarlaUE4.sh -RenderOffScreen

#srun echo 'srun podman run -d --name carla_container -p 2000-2002:2000-2002 docker.io/carlasim/carla:0.9.14 ./CarlaUE4.sh -RenderOffScreen'
#srun podman run -d --name carla_container -p 2000-2002:2000-2002 docker.io/carlasim/carla:0.9.14 ./CarlaUE4.sh -RenderOffScreen
#srun nvidia-smi

#singularity pull carla-0.9.14.sif docker://carlasim/carla:0.9.14
#srun --gpus=1 --pty singularity exec --nv /home/n/nsambhu/github/podman-carla/carla-0.9.14.sif /home/carla/CarlaUE4.sh -RenderOffScreen & nvidia-smi
# srun singularity exec --nv /home/n/nsambhu/github/podman-carla/carla-0.9.14.sif /home/carla/CarlaUE4.sh -RenderOffScreen &
# sleep 30
# nvidia-smi
# killall -9 -r CarlaUE4-Linux
# nvidia-smi

conda activate carla_py3.9
srun python -u run/2023_12_16_15parent_rl_custom.py
