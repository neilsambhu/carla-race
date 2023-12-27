import time, os, shutil, subprocess, glob, signal

bLocalCarla = False

# delete files from previous run
def clean_directory(directory):
    [os.remove(os.path.join(directory, file)) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    [shutil.rmtree(os.path.join(directory, dir)) for dir in os.listdir(directory) if os.path.isdir(os.path.join(directory, dir))]
clean_directory(directory='_out_16rl_custom2')
clean_directory(directory='tmp')
clean_directory(directory='models')
clean_directory(directory='logs')

start_time = time.time()
def print_elapsed_time():
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time_seconds = end_time - start_time
    # Convert seconds to hours, minutes, and seconds
    hours = int(elapsed_time_seconds // 3600)
    minutes = int((elapsed_time_seconds % 3600) // 60)
    seconds = int(elapsed_time_seconds % 60)
    # Display elapsed time in HH:MM:SS format
    print(f"Elapsed time: {hours:02}:{minutes:02}:{seconds:02}")

def kill_carla():
    kill_process = subprocess.Popen('killall -9 -r CarlaUE4-Linux', shell=True)
    kill_process.wait()
def kill_carla_remote():
    kill_process = subprocess.Popen('ssh SAMBHU23 "killall -9 -r CarlaUE4-Linux"', shell=True)
    kill_process.wait()

# check if saved final model exists
run = 1
while len(glob.glob('models/final.model')) == 0:
    print(f'Start run at count {run}')
    if bLocalCarla:
        kill_carla()
    else:
        kill_carla_remote()
    carla = None
    rl_custom = None
    try:
        if bLocalCarla:
            carla = subprocess.Popen('/opt/carla-simulator/CarlaUE4.sh -RenderOffScreen', shell=True, preexec_fn=os.setsid)
            # carla = subprocess.Popen('/opt/carla-simulator/CarlaUE4.sh', shell=True, preexec_fn=os.setsid)
        else:
            carla = subprocess.Popen('ssh SAMBHU23 "/opt/carla-simulator/CarlaUE4.sh -RenderOffScreen"', shell=True, preexec_fn=os.setsid)
            time.sleep(5)
        if run == 1:
            rl_custom = subprocess.Popen('python -u run/2023_12_18_16rl_custom2.py 2>&1 | tee out.txt', shell=True)
        elif run > 1:
            rl_custom = subprocess.Popen('python -u run/2023_12_18_16rl_custom2.py 2>&1 | tee -a out.txt', shell=True)
        def signal_handler(sig, frame):
            carla.terminate()
            rl_custom.terminate()
        signal.signal(signal.SIGINT, signal_handler)
        rl_custom.wait()
    except Exception as e:
        print(f'Run errored at count {run}')
        print(f'Parent error message: {e}')
        print(f'Continue to next attempt for run at count {run}')
        continue
    else:
        print(f'(Allegedly) no exception occurred for run at count {run}. CARLA Simulator may have crashed.')
    finally:
        print(f'End run at count {run}')
        print_elapsed_time()
        run += 1
        carla.terminate()
        rl_custom.terminate()
print('done');import sys; sys.exit()