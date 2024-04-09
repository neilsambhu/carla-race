import subprocess
town = subprocess.Popen(['python', 'run/2024_02_19_19town.py'])
town.wait()

# for speed in [80, 200, 210]:
# for speed in range(30,201,10):
# for speed in range(80,29,-10):
for speed in range(80,201,10):
    print(f'START SPEED {speed}')
    for steeringDenominator in range(25,201,25):
        print('--------------------------------------------------')    
        strRunLabel = f'run for (1) target speed {speed} and '+ \
            f'(2) steer denominator {steeringDenominator}'
        print(f'started run for {strRunLabel}')
        driveConstantSpeed = subprocess.Popen([
            'python', 'run/2024_03_21_25rl.py', 
            '--speed', str(speed),
            '--steerDenominator', str(steeringDenominator)
            ])
        # driveConstantSpeed.wait()
        output, error = driveConstantSpeed.communicate()
        if driveConstantSpeed.returncode != 0: 
            print(f'error during {strRunLabel}')
    print(f'END SPEED {speed}')