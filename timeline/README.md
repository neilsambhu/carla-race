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
10/26/2023 11:37 AM: find "waypoint"
```
grep -r -e "waypoint" |& tee ~/github/carla-race/outgrep.txt
```
results are from no_rendering_mode.py and synchronous_mode.py.  
10/26/2023 12:00 PM: I think I will need to modify the car spawn script to collect my own data. Steps: (1) spawn car; (2) collect RGB data; (3) implement basic agent.  
10/26/2023 12:04 PM: BasicAgent search in examples.
```
(carla-race) nsambhu@CSE001022:/opt/carla-simulator/PythonAPI/examples$ grep -r -e "BasicAgent" |& tee ~/github/carla-race/outgrep.txt
automatic_control.py:from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
automatic_control.py:            agent = BasicAgent(world.player, 30)
```
10/26/2023 5:22 PM: automatic_control.py works like manual_control.py with autopilot enabled by default. automatic_control.py calls basic_agent.py. Important function in basic_agent.py:
```
def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
    """
    Adds a specific plan to the agent.

        :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
        :param stop_waypoint_creation: stops the automatic random creation of waypoints
        :param clean_queue: resets the current agent's plan
    """
    self._local_planner.set_global_plan(
        plan,
        stop_waypoint_creation=stop_waypoint_creation,
        clean_queue=clean_queue
    )
```
10/26/2023 7:12 PM: automatic_control paramaters
```
python automatic_control.py --filter "vehicle.tesla.model3" --sync
```
10/26/2023 7:19 PM: find waypoints or landmarks in Town04.  
10/27/2023 12:08 PM: must not use autopilot for movement through Town04: autopilot is built into CARLA. 
10/27/2023 12:14 PM: how do I define the navigation through Town04 for my own agent?
10/27/2023 12:31 PM: the agent can hit custom waypoints. How do I find waypoints? Try to spawn a vehicle at specific waypoints in Town04.  
10/27/2023 12:46 PM: commands before reboot:
```
(carla-race) nsambhu@CSE001022:/opt/carla-simulator$ ./CarlaUE4.sh 
(carla-race) nsambhu@CSE001022:~/github/carla-race$ python run/2023_10_24_02car.py 
```
TODO: get RGB data from car.  
10/27/2023 2:00 PM: problem: I cannot set synchronous mode and have images saved to hard disk simulataneously. I probably need to configure a tick to have images saved.  
10/27/2023 2:18 PM: TODO: read examples/sensor_synchronization.py.  
10/27/2023 2:22 PM: 2023_10_24_02car.py cannot exist as synchronous mode and saving images to hard disk. Make a copy to 2023_10_27_03camera.py.  
10/27/2023 2:31 PM: What is the next step in getting the sensor data stored to hard disk? 
10/27/2023 7:43 PM: successfully saved RGB image from one waypoint.  
10/27/2023 7:56 PM: I have the waypoints. I have the RGB visual of the waypoints. I need to create an agent that can drive through waypoints.  
10/29/2023 3:42 PM: TODO: get full insight on CARLA Autopilot to see how it's driving.  
10/29/2023 3:58 PM: manual_control.py does not call any of the scripts in navigation/.  
10/29/2023 4:01 PM: find "set_autopilot"
```
grep -r -e "set_autopilot" |& tee ~/github/carla-race/outgrep.txt
```
There are only function calls and not function definitions.  
10/29/2023 4:06 PM: set_autopilot is part of the carla.Vehicle class.  
```
grep -r -e "Vehicle" |& tee ~/github/carla-race/outgrep.txt
```
10/30/2023 4:41 PM: grep cuts off printing. SAMBHU23 needs to be rebooted.  
10/31/2023 11:29 AM: the carla.Vehicle class is part of the Python API. I will go through the Python API directory structure to see where autpilot exists.  
10/31/2023 11:30 AM: the PythonAPI directory:  
```
PythonAPI  
    carla  
    examples  
    util  
```
Search order for autopilot code: carla, util, examples.  
10/31/2023 11:51 AM: the autopilot code is part of C++ code.  
10/31/2023 2:37 PM: search for "set_autopilot" in the following directories:  
```
CarlaUE4
Co-Simulation
Engine
HDMaps
Import
Plugins
PythonAPI
Tools
```
10/31/2023 2:38 PM: searching in CarlaUE4 for "set_autopilot"  
10/31/2023 2:44 PM: look for Autopilot in GitHub: https://github.com/carla-simulator/carla/issues?q=autopilot+sort%3Acreated-asc  
11/3/2023 4:39 PM: TODO: try to run basic_agent.py. Small step: search for calls to "basic_agent" in PythonAPI.  
```
grep -r -e "basic_agent" |& tee ~/github/carla-race/outgrep.txt
```
11/3/2023 4:41 PM: answer:
```
(carla-race) nsambhu@CSE001022:/opt/carla-simulator/PythonAPI/examples$ grep -r -e "basic_agent" |& tee ~/github/carla-race/outgrep.txt
automatic_control.py:from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
```
11/3/2023 7:32 PM: automatic_control.py parmaters
```
python automatic_control.py --sync --filter "vehicle.tesla.model3" --agent Basic
```
11/3/2023 7:41 PM: CARLA is not working properly. I will reboot SAMBHU23 as a sanity check. Items to verify: (1) print of 'a' stops and (2) vehicle drives.  
11/3/2023 7:43 PM: commands before reboot:
```
(carla-race) nsambhu@CSE001022:/opt/carla-simulator$ ./CarlaUE4.sh 
(carla-race) nsambhu@CSE001022:/opt/carla-simulator/PythonAPI/examples$ python automatic_control.py --sync --filter "vehicle.tesla.model3" --agent Basic
```
11/3/2023 8:07 PM: Neil TODO: read (1) automatic_control.py and (2) basic_agent.py.  
11/3/2023 10:14 PM: automatic_control.py: got to CollisionSensor.  
11/6/2023 2:00 PM: finished reading basic_agent.py.  
11/6/2023 2:06 PM: finished reading local_planner.py.  
11/6/2023 2:36 PM: copy API. New command for automatic_control.py
```
python run/PythonAPI/examples/automatic_control.py --sync --filter "vehicle.tesla.model3" --agent Basic
```
11/6/2023 2:47 PM: TODO: (1) turn off red lights and (2) print waypoints between 2 locations.  
11/6/2023 2:57 PM: ignore traffic lights set.  
11/6/2023 3:03 PM: reboot SAMBHU23 as a sanity check. Commands before reboot:
```
(carla-race) nsambhu@CSE001022:/opt/carla-simulator$ ./CarlaUE4.sh 
(carla-race) nsambhu@CSE001022:~/github/carla-race$ python run/PythonAPI/examples/automatic_control.py --sync --filter "vehicle.tesla.model3" --agent Basic
```
11/6/2023 4:39 PM: throttle modulation is choppy regardless of fixed_delta_seconds value.  
11/7/2023 9:50 AM: vehicle navigation between 2 nearby locations successfully completed.  
11/7/2023 10:22 AM: find low-level vehicle controls. Look for VehicleControl.  
11/7/2023 4:10 PM: there are no references to VehicleControl in basic_agent.py. I will check locations for low-level vehicle control.  
11/7/2023 5:49 PM: TODO: (1) find VehicleControl using grep and (2) create my own Autopilot by either (2.1) using basic_agent.py iteratively to navigate between waypoints or (2.2) controlling the low-level controls of the vehicle.  
```
grep -r -e "VehicleControl" --exclude=*.md --exclude=outgrep.txt|& tee ~/github/carla-race/outgrep.txt
```
11/9/2023 7:33 PM: find "brake"
```
grep -r -e "brake" --exclude=*.md --exclude=outgrep.txt|& tee ~/github/carla-race/outgrep.txt
```
11/10/2023 1:24 PM: TODO: look at controller.py.  
11/10/2023 4:41 PM: TODO: make ego vehicle drive between two waypoints.  
11/10/2023 4:49 PM: there is a pathway from basic_agent.py to controller.py.  
basic_agent.py
```
def __init__(self, vehicle, target_speed=20, opt_dict={}, map_inst=None, grp_inst=None):
```
```
self._local_planner = LocalPlanner(self._vehicle, opt_dict=opt_dict, map_inst=self._map)
```
local_planner.py
```
def _init_controller(self):
```
```
self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                args_lateral=self._args_lateral_dict,
                                                args_longitudinal=self._args_longitudinal_dict,
                                                offset=self._offset,
                                                max_throttle=self._max_throt,
                                                max_brake=self._max_brake,
                                                max_steering=self._max_steer)
```
controller.py
```
def run_step(self, target_speed, waypoint):
```
11/11/2023 6:59 PM: spawn command
```
python run/2023_11_06_05town.py && python -u run/PythonAPI/examples/automatic_control.py --sync --filter "vehicle.tesla.model3" --agent Basic |& tee out.txt
```
11/11/2023 7:07 PM: I was trying to set max_throttle to 1.0. I'm seeing where target_speed is referenced.  
11/12/2023 5:45 PM: I will find the choice of waypoints in the global_route_planner between the source and destination waypoints.  
11/12/2023 5:49 PM: GlobalRoutePlanner > trace_route  
11/12/2023 5:56 PM: macroscopic view: (1) waypoint navigation is more important to understand first than (2) maximizing speed.  
11/12/2023 6:26 PM: debug route
```
route = self._path_search(origin, destination)
```
```
route: [25, 239, 240, 180]
```
11/12/2023 6:33 PM: TODO: understand how `_path_search` works.  
11/12/2023 6:36 PM: `_path_search` returns int ids in graph `self._graph`.  
11/12/2023 6:38 PM: TODO: dnderstand initialization of `self._graph` in global_route_planner.py.  
11/12/2023 9:27 PM: `_build_graph` in global_route_planner.py builds the graph.  
11/12/2023 9:32 PM: `_build_graph` uses `self._topology`.  
11/12/2023 9:35 PM: TODO: find where `self._topology` is created.  
11/12/2023 9:36 PM: `_build_topology` creates `self._topology` using `self._wmap`.  
11/12/2023 9:44 PM: `self._wmap` is initialized in the GlobalRoutePlanner constructor. TODO: find the wmap object passed into the GlobalRoutePlanner constructor.  
11/12/2023 9:48 PM: basic_agent.py uses
```
self._map = self._world.get_map()
```
11/12/2023 10:06 PM: TODO: read documentation about maps.
```
BasicAgent self._map: Map(name=Carla/Maps/Town04_Opt)
```
11/12/2023 10:53 PM: see how `_build_topology` in global_route_planner.py references `self._wmap`.  
11/12/2023 10:56 PM: `_build_topology` calls the `get_topology()` method of wmap.
```
for segment in self._wmap.get_topology():
```
11/12/2023 11:07 PM: The `get_topology()` method of wmap returns a list of CARLA Waypoints.  
11/12/2023 11:18 PM: TODO: find out what `get_topology()` adds to the CARLA Waypoints.  
11/12/2023 11:25 PM: TODO: (1) read global_route_planner.py > `_build_topology()`.  
11/13/2023 10:07 AM: start research.  
11/13/2023 10:08 AM: Q1: what is the transform of the waypoint?  
11/13/2023 10:09 AM: A1: Transform contains location and rotation information.  
11/13/2023 10:10 AM: Location: line 110.  
11/13/2023 10:12 AM: Location: GlobalRoutePlanner > `_build_topology` creates a topology list by sequentially adding instances of `seg_dict`.  
11/13/2023 10:19 AM: Question: what is the purpose of `seg_dict`? What does `seg_dict` represent?  
11/13/2023 10:30 AM: GlobalRoutePlanner > `_build_topology()` > `_sampling_resolution`. TODO: print `sampling_resolution` to console.  
11/13/2023 10:39 AM: GlobalRoutePlanner has a sampling_resolution of 2.0. I don't know what this sampling resolution signifies.  
```
self._sampling_resolution = 2.0
```
11/13/2023 10:46 AM: see what CARLA 0.9.15 includes.  
11/13/2023 11:24 AM: sampling_resolution is unknown.  
11/13/2023 11:32 AM: `_sampling_resolution` is in the same units as the GPS coordinates.  
11/13/2023 11:36 AM: TODO: continue understanding how `_build_topology()`uses `seg_dict`.  
11/13/2023 11:42 AM: TODO: understand next method of waypoint.  
11/13/2023 11:56 AM: Summary: waypoint next method returns an array of one waypoint.  
11/13/2023 12:52 PM: Correction: The `get_topology()` method of wmap returns a list of paired CARLA Waypoints.  
11/13/2023 12:58 PM: Correction: find out what `get_topology()` adds to the topology list.  
11/13/2023 2:42 PM: `for segment in self._wmap.get_topology():`, find out what `seg_dict['path']` represents.  
11/13/2023 2:58 PM: `seg_dict['path']` represents the extrapolated path from the first node in the segment.  
11/13/2023 3:03 PM: find what `seg_dict` represents: 
```
for each waypoint pair, the path is extrapolated from the source until the path is close enough to the destination.  
```
11/13/2023 7:19 PM: find what `_topology` represents: list of `seg_dict` values.  
11/13/2023 7:27 PM: TODO: understand how `_build_graph` uses `_topology`.  
11/14/2023 12:36 PM: TODO: start understanding `_build_graph`. Later, understand the relevance of the graph. Later TODO: what does the graph represent in English words?  
11/14/2023 12:39 PM: Assumption: `_build_graph` will likely use `_topology`.  
11/14/2023 12:43 PM: TODO: find the English words representation of `_topology`: lookback.  
11/14/2023 12:47 PM: there is a limiting factor: 
```
the `next()` function of the waypoint automatically returns the correct path to the destination waypoint.  
```
11/14/2023 7:21 PM: `build_graph` creates a graph that represents:
```
a more detailed representation of what exists in the topology
```
11/18/2023 2:58 PM: print driving instructions and waypoint updates between source and destination.  
11/18/2023 7:46 PM: I have the list of waypoints in a route. I need to print the upcoming waypoints.  
11/20/2023 8:05 PM: Neil TODO: lookup table to replicate CARLA AP. Katkoori: waypoint and target speed. Katkoori: next loop, play with speed. Katkoori: see if there is RL package to import.  
11/22/2023 11:41 AM: computer is slow and needs to reboot.  
```
(carla-race) nsambhu@CSE001022:/opt/carla-simulator$ ./CarlaUE4.sh -RenderOffScreen
```
```
(carla-race) nsambhu@CSE001022:~/github/carla-race$ python run/2023_11_06_05town.py && python -u run/PythonAPI/examples/automatic_control.py --sync --filter "vehicle.tesla.model3" --agent Basic |& tee out.txt
```
11/22/2023 12:17 PM: error from control signal
```
__init__(_object*, float throttle=0.0, float steer=0.0, float brake=0.0, bool hand_brake=False, bool reverse=False, bool manual_gear_shift=False, int gear=0)
```
12/1/2023 7:20 PM:  
Katkoori: Neil TODO: collect GPS data to create overview of path of vehicle graphically. See if Omkar's dissertation includes this.  
Katkoori: Neil TODO: find simple pedagogical RL problem (e.g., tic-tac-toe).  
Katkoori: high-level overview: location-aware controls. Later, RL will give values.  
Katkoori: how can we define a basic loss function penalty of the car going off track?  
12/9/2023 4:35 PM: TODO: (1) update Anaconda; (2) update CARLA from 0.9.14 to 0.9.15.  
12/9/2023 4:37 PM: backup carla environment.  
12/9/2023 5:22 PM: Anaconda updated; CARLA 0.9.14 environment backed up. TODO: install CARLA 0.9.15 build from source.  
12/9/2023 6:31 PM: add to `~/.bashrc` default conda environment.  
```
# 2023 12 09 6:30 PM: initialize default conda environment
conda activate carla_0.9.15
```
12/9/2023 6:49 PM: install sandbox
```
sudo apt-add-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic main" &&
sudo apt-get update &&
sudo apt-get install build-essential clang-8 lld-8 g++-7 cmake ninja-build libvulkan1 python python-pip python-dev python3-dev python3-pip libpng-dev libtiff5-dev libjpeg-dev tzdata sed curl unzip autoconf libtool rsync libxml2-dev git &&
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/lib/llvm-8/bin/clang++ 180 &&
sudo update-alternatives --install /usr/bin/clang clang /usr/lib/llvm-8/bin/clang 180
```
12/9/2023 7:45 PM: rename branch for carla for NeilBranch
```
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git remote -v
origin  git@github.com:neilsambhu/carla.git (fetch)
origin  git@github.com:neilsambhu/carla.git (push)
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git remote rename origin custom
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git remote -v
custom  git@github.com:neilsambhu/carla.git (fetch)
custom  git@github.com:neilsambhu/carla.git (push)
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git remote add origin git@github.com:carla-simulator/carla.git
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git remote -v
custom  git@github.com:neilsambhu/carla.git (fetch)
custom  git@github.com:neilsambhu/carla.git (push)
origin  git@github.com:carla-simulator/carla.git (fetch)
origin  git@github.com:carla-simulator/carla.git (push)
```
12/9/2023 7:50 PM: git fetch from carla-simulator github
```
git fetch origin
```
```
 * [new tag]             0.9.15                                         -> 0.9.15
```
12/9/2023 7:56 PM: git transfer needs to be tag and not branch.  
```
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git tag
0.9.15
```
12/9/2023 7:58 PM: git checkout
```
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git checkout 0.9.15
Checking out files: 100% (2149/2149), done.
Note: checking out '0.9.15'.

You are in 'detached HEAD' state. You can look around, make experimental
changes and commit them, and you can discard any commits you make in this
state without impacting any branches by performing another checkout.

If you want to create a new branch to retain commits you create, you may
do so (now or later) by using -b with the checkout command again. Example:

  git checkout -b <new-branch-name>

HEAD is now at d7b45c1e1 Merge branch 'dev'
```
12/9/2023 8:10 PM: a tag is a specific type of branch without updates to history.  
12/9/2023 8:41 PM: git pull
```
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git pull origin master
From github.com:carla-simulator/carla
 * branch                master     -> FETCH_HEAD
Already up to date.
```
12/9/2023 8:47 PM: look for tag
```
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git tag -l "0.9.15*"
0.9.15
0.9.15-dev
```
12/9/2023 8:54 PM: switch to master branch
```
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git branch
* (HEAD detached at d7b45c1e1)
  NeilBranch0.9.13
  master
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git checkout master
Switched to branch 'master'
Your branch is up to date with 'origin/master'.
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git branch
  NeilBranch0.9.13
* master
```
12/9/2023 8:56 PM: I'm already on CARLA 0.9.15 repository. I need to push this to my own branch.  
```
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ head -10 CHANGELOG.md 
## CARLA 0.9.15
```
12/9/2023 9:02 PM: get new branch pushed to my GitHub
```
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git branch NeilBranch0.9.15
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git remote -v
custom  git@github.com:neilsambhu/carla.git (fetch)
custom  git@github.com:neilsambhu/carla.git (push)
origin  git@github.com:carla-simulator/carla.git (fetch)
origin  git@github.com:carla-simulator/carla.git (push)
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git status
On branch master
Your branch is up to date with 'origin/master'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)

    Unreal/CarlaUE4/.ignore
    outgrep.txt

nothing added to commit but untracked files present (use "git add" to track)
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git push custom
Counting objects: 7578, done.
Delta compression using up to 8 threads.
Compressing objects: 100% (3455/3455), done.
Writing objects: 100% (7578/7578), 990.99 MiB | 18.60 MiB/s, done.
Total 7578 (delta 4463), reused 6598 (delta 3509)
remote: Resolving deltas: 100% (4463/4463), completed with 408 local objects.
remote: warning: See https://gh.io/lfs for more information.
remote: warning: File Docs/img/tuto_content_authoring_vehicles/manual_control.gif is 52.17 MB; this is larger than GitHub's recommended maximum file size of 50.00 MB
remote: warning: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
remote: 
remote: Create a pull request for 'master' on GitHub by visiting:
remote:      https://github.com/neilsambhu/carla/pull/new/master
remote: 
To github.com:neilsambhu/carla.git
 * [new branch]          master -> master
```
12/9/2023 9:03 PM: new branch is called master and not NeilBranch0.9.15.  
12/9/2023 9:04 PM: TODO: (1) remove master branch from my carla GitHub; (2) push NeilBranch0.9.15 to my GitHub.  
```
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git branch
  NeilBranch0.9.13
  NeilBranch0.9.15
* master
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git checkout NeilBranch0.9.15
Switched to branch 'NeilBranch0.9.15'
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ git push custom
Counting objects: 7578, done.
Delta compression using up to 8 threads.
Compressing objects: 100% (3455/3455), done.
Writing objects: 100% (7578/7578), 991.03 MiB | 18.58 MiB/s, done.
Total 7578 (delta 4461), reused 6599 (delta 3509)
remote: Resolving deltas: 100% (4461/4461), completed with 408 local objects.
remote: warning: See https://gh.io/lfs for more information.
remote: warning: File Docs/img/tuto_content_authoring_vehicles/manual_control.gif is 52.17 MB; this is larger than GitHub's recommended maximum file size of 50.00 MB
remote: warning: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
remote: 
remote: Create a pull request for 'NeilBranch0.9.15' on GitHub by visiting:
remote:      https://github.com/neilsambhu/carla/pull/new/NeilBranch0.9.15
remote: 
To github.com:neilsambhu/carla.git
 * [new branch]          NeilBranch0.9.15 -> NeilBranch0.9.15
```
12/9/2023 9:24 PM: paused build at `https://carla.readthedocs.io/en/0.9.15/build_linux/` > Get assets  
12/9/2023 9:57 PM: message: `Content updated successfully.`
12/10/2023 1:38 PM: make is giving errors. `https://carla.readthedocs.io/en/0.9.15/build_linux/` > 1. Compile the Python API client. CLI bookmark
```
(carla_0.9.15) nsambhu@CSE001022:~/github/carla$ 
```
12/10/2023 1:51 PM: move UE4_ROOT farther down in `.bashrc`.  
12/10/2023 1:53 PM: make is still not working. Downgrade to CARLA 0.9.14.  
12/10/2023 2:02 PM: TODO: collect GPS data.  
12/10/2023 2:06 PM: TODO: get spawn data cleaned.  
12/10/2023 2:19 PM: spawn points in Town04 command
```
(carla_0.9.14_compiled) nsambhu@CSE001022:~/github/carla-race$ python -u run/2023_10_27_04spawn.py |& tee out.txt
```
12/10/2023 2:48 PM: spawn takes places in the air while car is coming down. Spawn points are a generaly indicator.  
12/10/2023 3:19 PM: spawn at index 335 is a good ending location for Town04. Index 0 is good for start.  
```
(carla_0.9.14_compiled) nsambhu@CSE001022:~/github/carla-race$ python -u run/2023_11_22_06spawn_check.py |& tee out.txt
```
12/10/2023 3:42 PM: CLI command for collecting GPS data 
```
python run/2023_11_06_05town.py && python -u run/PythonAPI/examples/automatic_control.py --sync --filter "vehicle.tesla.model3" --agent Basic |& tee out.txt
```
12/10/2023 3:50 PM: adding a delay to the ticks does not improve the smoothness of the drive.  
12/10/2023 3:56 PM: write components of vehicle location to text file.  
12/10/2023 5:10 PM: the delay to spawn the vehicle does work to prevent visuals of the vehicle falling immediately after spawning in the air.  
12/10/2023 8:39 PM: TODO: follow pytorch reinforcement learning DQN tutorial: `https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html`  
12/11/2023 3:24 PM: TODO: start RL model.  
12/12/2023 2:20 PM: YouTube source: `https://www.youtube.com/playlist?list=PLQVvvaa0QuDeI12McNQdnTlWz9XlCa0uo`
```
(carla_0.9.14_compiled) nsambhu@CSE001022:~/github/carla-race$ python run/2023_12_11_12tutorial.py 
```
12/12/2023 2:28 PM: for `run/2023_12_11_12tutorial.py`, there in a synchronization issue with showing the image.  
12/12/2023 3:30 PM: create second tutorial file.  
12/12/2023 10:44 PM: install cv2 and tensorflow
```
pip install opencv-python
```
```
pip install tensorflow-gpu==1.13.1
```
12/12/2023 10:46 PM: install Keras
```
pip install --upgrade keras
```
12/12/2023 10:50 PM: TODO: need to install CUDA Toolkit 10.0 and cudnn v7.5.0.
12/13/2023 1:10 AM: I need to install tensorflow 1.14.  
12/13/2023 2:33 AM: I need to install TensorFlow 1.9.0, before Keras became integrated into TensorFlow.  
12/13/2023 2:40 AM: TensorFlow 1.9.0 supports Python 3.6 at maximum.  
12/12/2023 3:10 AM: TODO: need to install CUDA Toolkit 11.0.  
12/13/2023 6:19 PM: need TensorFlow GPU. Remove nvida with purge command. 
```
sudo apt-get purge --auto-remove nvidia-cuda-toolkit
```
12/13/2023 6:21 PM: nvidia purge command
```
(carla_0.9.14_compiled) nsambhu@CSE001022:~/github/carla-race$ sudo apt-get purge --auto-remove nvidia-cuda-toolkit
[sudo] password for nsambhu: 
E: Could not get lock /var/lib/dpkg/lock-frontend - open (11: Resource temporarily unavailable)
E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), is another process using it?
```
12/13/2023 7:37 PM: (1) reboot, (2) uninstall cuDNN runtime and developer libraries, (3) uninstall CUDA 11.0, (4) install CUDA 12.1.  
12/14/2023 1:48 AM: tensorflow-cpu is installed. Error report:
```
(1152000,)
Traceback (most recent call last):
  File "run/2023_12_12_13tutorial.py", line 121, in <lambda>
    self.sensor.listen(lambda data: self.process_img(data))
  File "run/2023_12_12_13tutorial.py", line 145, in process_img
    i2 = i.reshape((self.im_height, self.im_width, 4))
ValueError: cannot reshape array of size 1152000 into shape (480,640,4)
```
line: `self.sensor.listen(lambda data: self.process_img(data))`
function:
```
def process_img(self, image):
    i = np.array(image.raw_data)
    print(i.shape)
    i2 = i.reshape((self.im_height, self.im_width, 4))
    i3 = i2[:, :, :3]
    if self.SHOW_CAM:
        cv2.imshow("", i3)
        cv2.waitKey(1)
    self.front_camera = i3
```
12/14/2023 2:19 AM: adjust image width from 640 to 600. The printout is accurate; CARLA.Image must be outputting the values switched.  
```
image.height: 600 image.width: 480
```
12/14/2023 3:07 AM: 16 minutes, 47 seconds to train for 100 episodes of 10 seconds, each. Base code is finished and runs.  
12/14/2023 3:14 AM: collect data of drives from base model.  
12/14/2023 3:23 AM: Offshoot: get NVIDIA drivers installed. No need to start trying to install CUDA Toolkit or cuDNN. Make sure the CUDA Toolkit version is compatible with Tensorflow 2.11.0.  
12/14/2023 4:08 AM: TODO: setup synchronous mode on script 14.  
12/14/2023 5:07 AM: Nvidia drivers are installed.  
12/14/2023 1:44 PM: synchronous mode on CARLA may or may not be working.  
12/14/2023 1:57 PM: TODO: remove threading.  
12/14/2023 1:59 PM: no need to remove threading. Threading is for the ML model.  
12/14/2023 2:11 PM: TODO: (1) pause synchronous mode and (2) save images to drive.  
12/14/2023 3:15 PM: images are saved to disk and grainy. Time delay is needed for write to hard disk. 
12/14/2023 3:20 PM: image is still grainy with delay. TODO: save carla image instead or processing raw data.  
12/14/2023 4:35 PM: TODO: switch height and width to match carla values.  
12/14/2023 7:12 PM: model is training. TODO: (1) spawn at 0 through 335 of Town 04; (2) modify reward function; (3) remove cap from episode length in units time.  
12/14/2023 7:30 PM: model converges on turning in a left circle.  
12/14/2023 7:50 PM: sync does not work on script 14. Changes in script 15 are (1) change town to Town04 and (2) set timeout to 60 seconds.  
12/14/2023 8:30 PM: code in script 14 is geting stuck and not able to proceed with tick.  
12/15/2023 12:59 AM: abandon sync for now. TODO: spawn Town04.  
12/15/2023 1:24 AM: sync not working is going to be a problem for obtaining the reward. Solution: I will collect the ground truth data without using sync.  
12/15/2023 1:46 AM: I need to get the code running without epsilon not being initialized.  
12/15/2023 3:21 PM: take main() code out to script file instead of within method.  
12/15/2023 3:27 PM: sync still doesn't work.  
12/15/2023 3:55 PM: CARLA crashes on episode 21.  
12/15/2023 4:00 PM: set timeout to 10 minutes.  
12/15/2023 4:20 PM: TODO: modify DQN network to train for more than 3 basic output control signals.  
12/15/2023 4:21 PM: CARLA crashed at episode 60.  
12/16/2023 2:03 PM: make large run() method.  
12/16/2023 2:06 PM: run() works. TODO: (1) save episode number for run; (2) start CARLA from command line.  
12/16/2023 2:17 PM: move try-except block to innermost function to see if I can catch the error and omit saving the variable state. I will print the episode number in the except block.  
12/16/2023 2:31 PM: try-except is not working. TODO: (1) remove try-except and (2) try to spawn CARLA programatically.  
12/16/2023 2:34 PM: try-except removed. Test error still appears (i.e., training should not reach 100 of 100 episodes).  
12/16/2023 2:39 PM: remove run() method.  
12/16/2023 2:43 PM: convolution error persists, where each step takes 1 second. TODO: reinstall Anaconda environment.  
12/16/2023 2:49 PM: Anaconda warning
```
==> WARNING: A newer version of conda exists. <==
  current version: 23.7.4
  latest version: 23.11.0

Please update conda by running

    $ conda update -n base -c defaults conda

Or to minimize the number of packages updated during conda update use

     conda install conda=23.11.0
```
12/16/2023 2:55 PM: convolution error persists. TODO: reboot.  
12/16/2023 3:10 PM: convolution error persists.  
12/16/2023 4:06 PM: code completes through 100 episodes.  
12/16/2023 4:11 PM: drive converges on turning in a left circle. TODO: modify DQN network to train for more than 3 basic output control signals.  
12/16/2023 5:32 PM: fail at episode 33.  
12/16/2023 5:37 PM: small step for TODO of more than 3 basic output control signals: 5 predictions.  
12/16/2023 5:47 PM: added actions 3 and 4
```
    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=True))
```
12/16/2023 7:18 PM: training works on 3 float values. Set brake to 0 for initial training.  
12/16/2023 8:23 PM: programmatically start and stop the simulator for each run.  
12/16/2023 8:57 PM: TODO: get tensorflow-gpu working.
12/16/2023 9:00 PM: reboot computer to see if carla simulator will stop erroring.  
12/16/2023 9:34 PM: TODO: need to get code to work from saved data.  
12/16/2023 11:34 PM: warning prevents code execution now
```
(carla_0.9.14_compiled) nsambhu@CSE001022:~/github/carla-race$ python -u run/2023_12_14_14rl_custom.py |& tee out.txt
1/1 [==============================] - 0s 122ms/step
0episodes [00:00, ?episodes/s]
WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _update_step_xla while saving (showing 2 of 2). These functions will not be directly callable after loading.
```
12/18/2023 2:51 AM: collect GPS data again from Town04 from spawn point 0 to spawn point 335. Inquiry to ChatGPT: 
```
I'm using CARLA Simulator 0.9.14. I used CARLA Autopilot to drive a vehicle.tesla.model3 from spawn point 0 to spawn point 335 in Town04 in synchronous mode with a fixed_delta_seconds of 0.05. I collected the x, y, z coordinates of the location of the car, stored in '_out_07vehicle_location_AP' separated by spaces and line returns. 
Write code in Python 3.7 to (1) open CARLA Simulator from /opt/carla-simulator/CarlaUE4.sh and train a reinforcement learning agent using these ground truth data. The action spaces exist as throttle from 0.0 to 1.0, steer from -1.0 to 1.0, and brake from 0.0 to 1.0. 
Vehicle controls are applied as follows:
throttle, steer, brake = action
vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake)))
Collect RGB image data from the vehicle stored to '_out_16rl_custom2/{episode_number}_{frame_number}.png'. 
```
12/18/2023 3:21 AM: ChatGPT response is too vague. I will refine the inquiry.
```
I'm using CARLA Simulator 0.9.14. I used CARLA Autopilot to drive a vehicle.tesla.model3 from spawn point 0 to spawn point 335 in Town04 in synchronous mode with a fixed_delta_seconds of 0.05. I collected the x, y, z coordinates of the location of the car, stored in '_out_07vehicle_location_AP/Town04_0_335.txt' separated by spaces and line returns. Distance is calculated as follows in CARLA Simulator: the carla.Location class takes in as parameters x, y, and z float values; the carla.Location.distance() method takes as input a carla.location object and returns the Euclidean distance from this location to another one. The action spaces exist as throttle from 0.0 to 1.0, steer from -1.0 to 1.0, and brake from 0.0 to 1.0. Vehicle controls are applied as follows:
throttle, steer, brake = action
vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake)))
Write each line of code in Python 3.7 to train a reinforcement learning agent rewarded by minimizing the distance from these ground truth location data for each tick in the CARLA Simulator. I already have the code written to spawn the vehicle at spawn point 0 in Town04. I need the code that minimizes the distance from the ground truth data.
```
12/18/2023 3:42 AM: best code output so far from ChatGPT:
```
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random

# Load data from the text file
data_file = 'collected_data.txt'  # Replace with your file path
with open(data_file, 'r') as file:
    data = file.read().splitlines()

# Preprocess the data
parsed_data = [list(map(float, line.split())) for line in data]
states = np.array(parsed_data)  # States: x, y, z coordinates

# Define action space
action_space = {'throttle': np.linspace(0.0, 1.0, num=10),
                'steer': np.linspace(-1.0, 1.0, num=10),
                'brake': np.linspace(0.0, 1.0, num=10)}

# Define DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay memory
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(action_space)  # Exploration
        return np.argmax(self.model.predict(state))  # Exploitation

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Initialize environment and agent
state_size = len(states[0])
action_size = len(action_space)
agent = DQNAgent(state_size, action_size)

# Training the agent
batch_size = 32
for episode in range(EPISODES):
    # Reset environment for each episode
    state = states[0]
    for time_step in range(len(states)):
        action = agent.act(state)
        next_state = states[time_step]
        reward = 1  # Define your reward function based on state transition
        done = True if time_step == len(states) - 1 else False
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
```
12/18/2023 4:02 AM: TODO: discretize the values of throttle, steer, and brake.  
12/18/2023 4:46 AM: found mistake in image_size_x for setting the camera attribute.  
12/18/2023 5:05 AM: useful code for mapping
```
class CarEnv:
    # Existing code...

    num_actions = 24  # Total number of actions

    def step(self, action):
        throttle_action = action // 12  # Values in range [0, 1]
        steer_action = (action % 12) // 4  # Values in range [0, 2]
        brake_action = action % 4  # Values in range [0, 3]

        throttle_value = self.action_space['throttle'][throttle_action]
        steer_value = self.action_space['steer'][steer_action]
        brake_value = self.action_space['brake'][brake_action]

        self.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle_value), steer=float(steer_value), brake=float(brake_value)))

        # Rest of your code...
```
12/20/2023 4:48 PM: for ground truth location data from CARLA AP run, `local_planner.py` collects the location datum before applying vehicle controls.  
12/20/2023 8:27 PM: get tensorflow-gpu working.  
12/20/2023 9:02 PM: TensorFlow needs CUDA Toolkit 11.0.  
12/20/2023 9:56 PM: Tensorflow needs cuDNN 7.  
12/20/2023 10:05 PM: TensorFlow needs CUDA Toolkit 10.2 to be compatible with cuDNN 7.  
12/20/2023 10:32 PM: upgrade CARLA and python versions.  
12/20/2023 11:53 PM: coordinate ticks and line numbers for ground truth AP run.  
12/25/2023 3:00 AM: TODO: make an IL model from the throttle values of the ground truth run.  
12/25/2023 11:58 PM: new CARLA Simulator command to recover after crash.
```
while true; do ./CarlaUE4.sh; done
```
12/26/2023 2:55 PM: TODO: reward fast driving without crashing.  
12/26/2023 5:10 PM: SAMBHU23 with CARLA Simulator crashed. TODO: install CARLA Simulator on SAMBHU24.  
12/26/2023 9:52 PM: new CARLA Simulator command to recover after crash; kill old process.
```
while true; do ./CarlaUE4.sh; [ $? -ne 0 ] && pkill -f CarlaUE4.sh; done
```
1/5/2024 2:52 PM: TODO: (1) check CARLA runs programmatically from python script; (2) make map of drive; (3) refine model
1/5/2023 11:23 PM: mount GAIVI directory for image files
```
sshfs nsambhu@gaivi.cse.usf.edu:/home/n/nsambhu ~/gaivi
```
1/5/2023 11:32 PM: TODO: (1) make map of drive; (2) refine ground truth data collection (i.e., no AP, allow spawn to finish, enable AP, output control signals).
1/6/2024 1:46 AM: max allocation of 8 GPUs.  
1/6/2024 10:10 PM: view output
```
tail -f slurm-143270.out
```
1/6/2024 10:19 PM: TODO: (1) monitor episode to be able to drive without crashing; (2) make map of drive; (3) refine ground truth drive data collection (i..e, no AP, allow spawn to finish, enable AP, output control).  