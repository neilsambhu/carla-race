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
10/24/2023 1:01 PM: TODO: spawn vehicle: choose vehicle to spawn: Tesla Model 3.  
10/24/2023 1:06 PM: TODO: decide if vehicle should be spawned through API or through manual_control.py window.  
10/24/2023 1:17 PM: TODO: get waypoint locations on map.  
10/24/2023 4:29 PM: TODO: review basic_agent.py to find waypoints.  
10/24/2023 4:31 PM: TODO: spawn vehicle through manual_control.py window.  
10/24/2023 5:23 PM: I changed the settings for asynchronous mode to synchronous mode. manual_control.py still reads asynchronous mode.  
10/24/2023 5:42 PM: I need to choose between (1) getting synchronous mode working and (2) understanding waypoints. I will reduce the rendering quality to get CARLA working and move on to the Autopilot waypoints.  
10/24/2023 7:53 PM: in the manual_control.py, find out how autopilot works.  
10/25/2023 10:38 AM: TODO: set synchronous mode as a parameter in manual_control.py.  
```
python manual_control.py --autopilot --filter "vehicle.tesla.model3" --sync
```
10/25/2023 10:41 AM: CARLA run
```
./CarlaUE4.sh -quality-level=Low -RenderOffScreen
```
10/25/2023 10:44 AM: reminder to myself: even though sync settings are set in manual_control.py, I will have to configure the waypoints manually. For example, I can modify the manual_control.py script myself.  
10/25/2023 10:53 AM: synchronous_mode and fixed_delta_seconds are set accurately.  
```
Python 3.7.13 (default, Mar 29 2022, 02:18:16) 
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import carla
>>> client = carla.Client('localhost', 2000)
>>> client.set_timeout(20.0)
>>> world = client.get_world()
>>> world.get_settings()
<carla.libcarla.WorldSettings object at 0x7f4db4cf4ae0>
>>> world.get_settings().synchronous_mode
True
>>> world.get_settings().fixed_delta_seconds
0.05
>>> 
```
10/25/2023 11:00 AM: TODO: find waypoints in manual_control.py. There might be a call from manual_control.py to basic_agent.py.  
10/25/2023 11:18 AM: TODO: find how autopilot navigates in manal_control.py.  
10/25/2023 11:23 AM: I cannot find where the autopilot calls low-level functions from manual_control.py.  
10/25/2023 1:03 PM: `manual_control.py` > autopilot start
```
controller = KeyboardControl(world, args.autopilot)
```
10/25/2023 3:06 PM: `manual_control.py` > class KeyboardControl
```
if controller.parse_events(client, world, clock, args.sync):
                return
```
10/25/2023 3:34 PM: fine-grained Autopilot settings are probably not in the KeyboardControl class.  
10/25/2023 3:44 PM: fine-grained Autopilot settings might be in the World class.  
10/25/2023 4:25 PM: read traffic manager: "The Traffic Manager (TM) is the module that controls vehicles in autopilot mode in a simulation."  
10/25/2023 7:23 PM: manual_control.py > game_loop(args)
```
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)
```
10/25/2023 7:27 PM: autopilot settings might be in world class. 
```
world.player.set_autopilot(self._autopilot_enabled)
```
10/25/2023 7:30 PM: player location
```
self.player = self.world.try_spawn_actor(blueprint, spawn_point)
```
10/25/2023 7:42 PM: the search for the autopilot waypoints is not obviously in the traffic_manager nor world.player.  
10/26/2023 11:17 AM: read more about traffic manager.  