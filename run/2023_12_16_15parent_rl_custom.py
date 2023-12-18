import time, os, shutil, subprocess, glob

# delete files from previous run
def clean_directory(directory):
    [os.remove(os.path.join(directory, file)) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    [shutil.rmtree(os.path.join(directory, dir)) for dir in os.listdir(directory) if os.path.isdir(os.path.join(directory, dir))]
clean_directory(directory='_out_16rl_custom2')
clean_directory(directory='tmp')
clean_directory(directory='models')

def kill_carla():
    kill_process = subprocess.Popen('killall -9 -r CarlaUE4-Linux', shell=True)
    kill_process.wait()

# check if saved final model exists
run = 1
while len(glob.glob('models/final.model')) == 0:
    kill_carla()
    carla = None
    rl_custom = None
    try:
        carla = subprocess.Popen('/opt/carla-simulator/CarlaUE4.sh -RenderOffScreen', shell=True, preexec_fn=os.setsid)
        if run == 1:
            rl_custom = subprocess.Popen('python -u run/2023_12_16_15rl_custom.py 2>&1 | tee out.txt', shell=True)
        elif run > 1:
            rl_custom = subprocess.Popen('python -u run/2023_12_16_15rl_custom.py 2>&1 | tee -a out.txt', shell=True)
        carla.wait()
        rl_custom.wait()
    except Exception as e:
        print(f'Run errored at count {run}')
        print(f'Parent error message: {e}')
        print(f'Continue to next attempt for run at count {run}')
        continue
    else:
        print(f'(Allegedly) no exception occurred for run at count {run}. CARLA Simulator may have crashed.')
    finally:
        run += 1
        carla.terminate()
        rl_custom.terminate()
print('done')