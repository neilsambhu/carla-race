import carla, time, queue, shutil, os, glob, math, configparser

config = configparser.ConfigParser()
config.read('config.ini')
bSAMBHU23 = config.getboolean('Settings','bSAMBHU23')
bGPU_random = config.getboolean('Settings','bGPU_random')
bGAIVI = not bSAMBHU23

path_AP_controls = '_out_21_CARLA_AP_Town06/Controls.txt'
path_AP_locations = '_out_21_CARLA_AP_Town06/Locations.txt'

dir_outptut = '_out_22_rl'
if os.path.exists(dir_outptut):
    shutil.rmtree(dir_outptut)
os.makedirs(dir_outptut)
dir_output_frames = f'{dir_outptut}/frames/'
path_rl_controls = f'{dir_outptut}/Controls.txt'
path_rl_locations = f'{dir_outptut}/Locations.txt'

'''Make sure CARLA Simulator 0.9.14 is running'''
actor_list = []

def actor_list_destroy(actor_list):
    [x.destroy() for x in actor_list]
    return []
def getPath_CARLA_AP_Town06():
    listLocationsPath = []
    with open(path_AP_locations,'r') as file_AP_locations:
        for line in file_AP_locations.readlines():
            lineStripped = line.strip()
            x,y,z = lineStripped.split()
            locationFromPath = carla.Location(float(x),float(y),float(z))
            listLocationsPath.append(locationFromPath)
    return listLocationsPath
listLocationsPath = getPath_CARLA_AP_Town06()
def getLocationClosestToCurrent(currentLocation):
    distanceMinimum = None
    listDistance = []
    for locationFromPath in listLocationsPath:
        distanceFromPath = currentLocation.distance(locationFromPath)
        listDistance.append(distanceFromPath)
    distanceMinimum = min(listDistance)
    indexMinimum = listDistance.index(distanceMinimum)
    return listLocationsPath[indexMinimum]

def main():
    try:
        # Connect to the CARLA Simulator
        if bSAMBHU23:
            client = carla.Client('localhost', 2000)
            client.set_timeout(20.0)
        if bGAIVI:
            command_output = subprocess.run(['squeue'], capture_output=True, text=True)
            output_lines = command_output.stdout.split('\n')
            carla_line = [line for line in output_lines if 'nsambhu' in line and 'carla.sh' in line and 'GPU' in line]
            gpu_info = carla_line[-1].split()[-1]  # Assuming GPU info is the last column
            print("GPU Info for carla.sh:", gpu_info)
            self.client = carla.Client(gpu_info, 2000)
            self.client.set_timeout(120)

        # Get the world object
        world = client.get_world()
        world = client.load_world('Town06_Opt')

        # Set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True # Enables synchronous mode
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Define the blueprint of the vehicle you want to spawn
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

        # Now we need to give an initial transform to the vehicle. We choose a
        # random transform from the list of recommended spawn points of the map.
        height = 0.1
        spawn_start_left = carla.Transform(carla.Location(x=19.7, y=240.9, z=height), carla.Rotation())
        spawn_start_center = carla.Transform(carla.Location(x=19.7, y=244.4, z=height), carla.Rotation())
        spawn_start_right = carla.Transform(carla.Location(x=19.7, y=247.9, z=height), carla.Rotation())
        location_destination = carla.Location(x=581.2, y=244.6, z=height)
        # transform = spawn_start_center
        transform = spawn_start_left

        # So let's tell the world to spawn the vehicle.
        vehicle = world.spawn_actor(vehicle_bp, transform)

        # It is important to note that the actors we create won't be destroyed
        # unless we call their "destroy" function. If we fail to call "destroy"
        # they will stay in the simulation even after we quit the Python script.
        # For that reason, we are storing all the actors we create so we can
        # destroy them afterwards.
        actor_list.append(vehicle)

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(False)

        # Let's add now a "depth" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)

        # Now we register the function that will be called each time the sensor
        # receives an image. In this example we are saving the image to disk.
        camera.listen(lambda image: image.save_to_disk(f'{dir_output_frames}/%06d.png' % image.frame))

        countTick = 0
        world.tick()
        countTick += 1
        def getDistanceToDestination():
            return location_destination.distance(vehicle.get_location())
        throttle, steer, brake = 1.0, 0.0, 0.0
        while getDistanceToDestination() > 2:
            locationClosestToCurrent = getLocationClosestToCurrent(vehicle.get_location())
            deltaY = locationClosestToCurrent.y - vehicle.get_location().y
            thresholdDeltaY = 1
            bWithinThreshold = None
            unitChangeThrottle = 0.1
            unitChangeSteer = 0.1
            unitChangeBrake = 0.1
            if deltaY >= -thresholdDeltaY and deltaY <= thresholdDeltaY:
                bWithinThreshold = True
                throttle, steer, brake = 1.0, 0.0, 0.0
            elif deltaY < -thresholdDeltaY:
                bWithinThreshold = False
                deltaSteer = -unitChangeSteer
                steer = max(steer+deltaSteer, -0.1)
            elif deltaY > thresholdDeltaY:
                bWithinThreshold = False
                deltaSteer = unitChangeSteer
                steer = min(steer+deltaSteer, 0.1)
            if not bWithinThreshold:
                v = vehicle.get_velocity()
                kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
                if kmh < 5:
                    # slow or not moving
                    brake = 0
                    deltaThrottle = unitChangeThrottle
                    throttle = min(throttle+deltaThrottle, 1.0)
                else:
                    # already moving
                    throttle = 0.0
                    deltaBrake = unitChangeBrake
                    brake = min(brake+deltaBrake, 1.0)
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
            print(f'Distance to destination: {getDistanceToDestination():.1f}\tdeltaY: {deltaY:.2f}\tthrottle: {throttle:.1f} steer: {steer:.1f} brake: {brake:.1f}')
            world.tick()
            countTick += 1
        countTick -= 1
        while len(glob.glob(os.path.join(dir_output_frames,'*'))) < countTick:
            time.sleep(10)
    finally:
        actor_list_destroy(actor_list)
        print('done')

if __name__ == '__main__':
    main()