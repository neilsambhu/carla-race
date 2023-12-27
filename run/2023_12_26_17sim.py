import os, subprocess

# carla = subprocess.Popen('/opt/carla-simulator/CarlaUE4.sh -RenderOffScreen', shell=True, preexec_fn=os.setsid)
carla = subprocess.Popen('/opt/carla-simulator/CarlaUE4.sh', shell=True, preexec_fn=os.setsid)
