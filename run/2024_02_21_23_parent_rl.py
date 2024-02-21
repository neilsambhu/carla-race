import time, os, shutil, subprocess, glob, signal, configparser

config = configparser.ConfigParser()
config.read('config.ini')
bSAMBHU23 = config.getboolean('Settings','bSAMBHU23')
bLocalCarla = not bSAMBHU23
bGAIVI = not bSAMBHU23
# count_max_runs = 1
count_max_runs = int(1e6)
pathPythonScript = 'run/2024_02_19_22rl.py'
# delete files from previous run
def clean_directory(directory):
    if not bGAIVI:
        [os.remove(os.path.join(directory, file)) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
        [shutil.rmtree(os.path.join(directory, dir)) for dir in os.listdir(directory) if os.path.isdir(os.path.join(directory, dir))]
    else:
        clean = subprocess.Popen(f'rm -rf {directory}/*', shell=True)
        clean.wait()

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
def kill_carla_gaivi():
    kill_process = subprocess.Popen('scancel --name="carla.sh"', shell=True)
    kill_process.wait()
    # time.sleep(10)

# check if saved final model exists
run = 1

if bLocalCarla:
    if not bGAIVI:
        kill_carla()
    else:
        # kill_carla_gaivi()
        pass
else:
    kill_carla_remote()
carla = None
rl_custom = None
try:
    if bLocalCarla:
        if not bGAIVI:
            carla = subprocess.Popen('/opt/carla-simulator/CarlaUE4.sh -RenderOffScreen', shell=True, preexec_fn=os.setsid)
            # carla = subprocess.Popen('/opt/carla-simulator/CarlaUE4.sh', shell=True, preexec_fn=os.setsid)
        else:
            print(f"before carla, run squeue")
            squeue_before_carla = subprocess.Popen('squeue | grep nsambhu', shell=True)
            squeue_before_carla.wait()
            # import random
            # time.sleep(random.uniform(0,60))
            # if existing CARLA is present, don't spawn CARLA
            carla_line = []
            command_output = subprocess.run(['squeue'], capture_output=True, text=True)
            output_lines = command_output.stdout.split('\n')
            carla_line = [line for line in output_lines if 'nsambhu' in line and 'carla.sh' in line and not 'CG' in line]
            if len(carla_line) == 0:
                carla = subprocess.Popen(f'sbatch /home/n/nsambhu/github/podman-carla/carla.sh ', shell=True)
                carla.wait()
                time.sleep(10)
            print(f"after carla, run squeue")
            squeue_after_carla = subprocess.Popen('squeue | grep nsambhu', shell=True)
            squeue_after_carla.wait()
            carla_line = []
            while carla_line == []:
                command_output = subprocess.run(['squeue'], capture_output=True, text=True)
                output_lines = command_output.stdout.split('\n')
                carla_line = [line for line in output_lines if 'nsambhu' in line and 'carla.sh' in line and 'GPU' in line and not 'CG' in line]
            print(f'carla_line: {carla_line}')
            carla_gpu_info = carla_line[-1].split()[-1]  # Assuming GPU info is the last column
            print("GPU Info for carla.sh:", carla_gpu_info)
            import carla
            # time.sleep(60)
            world = None
            count_attempt = 0
            count_max_attempt = 10
            while world == None and count_attempt<count_max_attempt:
                try:
                    client = carla.Client(str(carla_gpu_info), 2000)
                    world = client.get_world()
                    print(f'world: {world}')
                except Exception as e:
                    count_attempt+=1
                    print(f'CARLA starting error message: {e}')
                    time.sleep(1)
                    continue
            if world == None:
                raise Exception('Code could not connect to CARLA Simulator.')
            # time.sleep(30)                
            # nvidia_smi = subprocess.Popen('nvidia-smi', shell=True, preexec_fn=os.setsid)
            # nvidia_smi.wait()
    else:
        carla = subprocess.Popen('ssh SAMBHU23 "/opt/carla-simulator/CarlaUE4.sh -RenderOffScreen"', shell=True, preexec_fn=os.setsid)
        time.sleep(30)
    if run == 1:
        if bSAMBHU23:
            rl_custom = subprocess.Popen(f'python -u {pathPythonScript} 2>&1 | tee out.txt', shell=True)
        elif bGAIVI:
            rl_custom = subprocess.Popen(f'python -u {pathPythonScript}', shell=True)
    elif run > 1:
        rl_custom = subprocess.Popen(f'python -u {pathPythonScript}', shell=True)
    def signal_handler(sig, frame):
        carla.terminate()
        rl_custom.terminate()
    signal.signal(signal.SIGINT, signal_handler)
    rl_custom.wait()
except Exception as e:
    print(f'Run errored at count {run}')
    print(f'Parent error message: {e}')
    print(f'Continue to next attempt for run at count {run}')
else:
    print(f'(Allegedly) no exception occurred for run at count {run}. CARLA Simulator may have crashed.')
finally:
    print(f'End run at count {run}')
    print_elapsed_time()
    run += 1
    if bSAMBHU23:
        carla.terminate()
    elif bGAIVI:
        kill_carla_gaivi()
    rl_custom.terminate()
print('done');import sys; sys.exit()