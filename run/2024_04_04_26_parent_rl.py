import subprocess
town = subprocess.Popen(['python', 'run/2024_02_19_19town.py'])
town.wait()

# for speed in [80, 200, 210]:
for speed in range(30,201,10):
    print('--------------------------------------------------')    
    print(f'started run for target speed {speed} km/h')
    driveConstantSpeed = subprocess.Popen(['python', 'run/2024_03_21_25rl.py', '--speed', str(speed)])
    # driveConstantSpeed.wait()
    output, error = driveConstantSpeed.communicate()
    if driveConstantSpeed.returncode != 0: 
        print(f'error during run for target speed {speed}')
        break