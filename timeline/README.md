# Notes for CARLA Race
10/23/2023 4:58:10 PM: TODO: find python version for CARLA 0.9.14: Python 3.7.  
10/23/2023 5:02:39 PM: TODO: setup conda environment for carla-race.  
```
conda create --name carla-race python=3.7
```
10/24/2023 10:34:02 AM: install CARLA 0.9.14 on SAMBHU23.  
10/24/2023 11:14:58 AM: after importing only the AdditionalMaps tar file, the following error exists when running CARLA:
```
(carla-race) nsambhu@CSE001022:/opt/carla-simulator$ ./CarlaUE4.sh 
4.26.2-0+++UE4+Release-4.26 522 0
Disabling core dumps.
Failed to find symbol file, expected location:
"/opt/carla-simulator/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping.sym"
LowLevelFatalError [File:Unknown] [Line: 3146] 
Could not find SuperStruct CarlaWheeledVehicleNW to create BaseVehiclePawnNW_C
Signal 11 caught.
Malloc Size=65538 LargeMemoryPoolOffset=65554 
CommonUnixCrashHandler: Signal=11
Malloc Size=131160 LargeMemoryPoolOffset=196744 
Malloc Size=131160 LargeMemoryPoolOffset=327928 
Engine crash handling finished; re-raising signal 11 for the default handler. Good bye.
Segmentation fault (core dumped)
```
TODO: download the other 2 tar files.  
10/24/2023 12:21:53 PM: CARLA 0.9.14 installed.  
10/24/2023 12:22:25 PM: TODO: configure CARLA 0.9.14 Town04.  