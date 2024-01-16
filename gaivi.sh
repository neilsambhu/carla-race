#!/bin/bash -l
#All options below are recommended
#SBATCH -p Contributors #general # run on partition general
#SBATCH --cpus-per-task=32 # 32 CPUs per task
#SBATCH --mem=100GB # 100GB per task
##SBATCH --mem=257264 # 100GB per task
#SBATCH --gpus=1 # 4 GPUs
##SBATCH --mail-user=nsambhu@mail.usf.edu # email for notifications
##SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE # events for notifications
#SBATCH -w GPU45

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
