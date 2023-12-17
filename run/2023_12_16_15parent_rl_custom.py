import time
# delete files from previous run
def delete_files(directory):
    import os
    if os.path.exists(directory):
        [os.remove(os.path.join(directory, file)) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    else:
        print("Directory does not exist or is already removed.")
delete_files(directory='_out_14rl_custom')
delete_files(directory='tmp')
delete_files(directory='models')

# check if saved final model exists
import glob
run = 1
while len(glob.glob('models/final.model')) == 0:
    import subprocess
    carla = subprocess.Popen('/opt/carla-simulator/CarlaUE4.sh', shell=True)
    time.sleep(10)
    try:
        rl_custom = subprocess.Popen('python -u run/2023_12_14_14rl_custom.py > out.txt', shell=True)
        time.sleep(10)
    except Exception as e:
        print(f'Run errored at count {run}')
        print(f'Parent error message: {e}')
        print(f'Continue to next attempt')
        run += 1
        continue
    else:
        print(f'No exception occurred for run at count {run}.')
    finally:
        carla.terminate()
        rl_custom.terminate()
    time.sleep(10)
