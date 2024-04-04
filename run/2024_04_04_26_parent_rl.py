import os
for speed in range(30,150,10):
    os.system(f'python run/2024_02_19_19town.py && python run/2024_03_21_25rl.py --speed {speed}')