import carla, time, queue, shutil, os, glob, math, configparser, subprocess, cv2
import numpy as np
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('config.ini')
bSAMBHU23 = config.getboolean('Settings','bSAMBHU23')
bGAIVI = not bSAMBHU23

path_AP_controls = '_out_21_CARLA_AP_Town06/ControlsStraight.txt'
path_AP_locations = '_out_21_CARLA_AP_Town06/LocationsStraight.txt'

dir_outptut = '_out_22_rl'
if not os.path.exists(dir_outptut):
    os.makedirs(dir_outptut)
dir_output_frames = f'{dir_outptut}/frames/'
if not os.path.exists(dir_output_frames):
    os.makedirs(dir_output_frames)

def clean_directory(directory):
    if not bGAIVI:
        [os.remove(os.path.join(directory, file)) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
        [shutil.rmtree(os.path.join(directory, dir)) for dir in os.listdir(directory) if os.path.isdir(os.path.join(directory, dir))]
    else:
        clean = subprocess.Popen(f'rm -rf {directory}/*', shell=True)
        clean.wait()
clean_directory(dir_output_frames)

path_rl_controls = f'{dir_outptut}/Controls.txt'
path_rl_locations = f'{dir_outptut}/Locations.txt'

'''Make sure CARLA Simulator 0.9.14 is running'''
actor_list = []
IM_WIDTH = 80*2
IM_HEIGHT = 60*2

def actor_list_destroy(actor_list):
    [x.destroy() for x in actor_list]
    return []
def getPath_CARLA_AP_Town06():
    listLocationsPath_CARLA_AP_Town06 = []
    with open(path_AP_locations,'r') as file_AP_locations:
        for line in file_AP_locations.readlines():
            lineStripped = line.strip()
            x,y,z = lineStripped.split()
            locationFromPath = carla.Location(float(x),float(y),float(z))
            listLocationsPath_CARLA_AP_Town06.append(locationFromPath)
    return listLocationsPath_CARLA_AP_Town06
listLocationsPath_CARLA_AP_Town06 = getPath_CARLA_AP_Town06()
def getLocationClosestToCurrent(currentLocation):
    distanceMinimum = None
    listDistance = []
    for locationFromPath in listLocationsPath_CARLA_AP_Town06:
        distanceFromPath = currentLocation.distance(locationFromPath)
        listDistance.append(distanceFromPath)
    distanceMinimum = min(listDistance)
    indexMinimum = listDistance.index(distanceMinimum)
    return listLocationsPath_CARLA_AP_Town06[indexMinimum]

def main():
    try:
        # Connect to the CARLA Simulator
        if bSAMBHU23:
            client = carla.Client('localhost', 2000)
            client.set_timeout(120.0)
        if bGAIVI:
            command_output = subprocess.run(['squeue'], capture_output=True, text=True)
            output_lines = command_output.stdout.split('\n')
            carla_line = [line for line in output_lines if 'nsambhu' in line and 'carla.sh' in line and 'GPU' in line]
            gpu_info = carla_line[-1].split()[-1]  # Assuming GPU info is the last column
            print("GPU Info for carla.sh:", gpu_info)
            client = carla.Client(gpu_info, 2000)
            client.set_timeout(120)

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
        camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        camera_bp.set_attribute("fov", f"110")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)

        # Now we register the function that will be called each time the sensor
        # receives an image. In this example we are saving the image to disk.
        # camera.listen(lambda image: image.save_to_disk(f'{dir_output_frames}/%06d.png' % image.frame))
        countTick = 0
        def processImage(image):
            i = np.array(image.raw_data)
            # print(i.shape)
            i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
            i3 = cv2.cvtColor(i2, cv2.COLOR_BGRA2RGB)
            from PIL import Image
            i4 = Image.fromarray(i3)
            # i4.save(os.path.join(dir_output_frames, f'{image.frame:06d}.png'))
            i4.save(os.path.join(dir_output_frames, f'{countTick:06d}.png'))
        camera.listen(lambda image: processImage(image))

        world.tick()
        countTick += 1
        def getDistanceToDestination():
            return location_destination.distance(vehicle.get_location())
        def getStandardVehicleControl():
            return 0.75, 0.0, 0.0
        throttle, steer, brake = getStandardVehicleControl()
        listDeltaY = []
        listLocations = []
        # Plot setup for delta Y
        fig_deltaY, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Delta Y')
        ax1.set_title('Delta Y over Time')
        # Plot setup for overlay
        fig_overlay, ax2 = plt.subplots(figsize=(12, 6))  # Adjust the figsize as needed
        ax2.set_aspect('equal', 'box')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Vehicle Location and Path Overlay')
        while getDistanceToDestination() > 10:
            locationClosestToCurrent = getLocationClosestToCurrent(vehicle.get_location())
            deltaY = vehicle.get_location().y - locationClosestToCurrent.y
            listDeltaY.append(deltaY)
            listLocations.append(vehicle.get_location())
            thresholdDeltaYNoSteer = 0.1
            thresholdDeltaYSteer = 2.5
            speedMinimum = 10
            speedTarget = 30
            bWithinThreshold = None
            maxSteer = None
            unitChangeThrottle = 0.1
            unitChangeSteer = 1
            unitChangeBrake = 0.1
            v = vehicle.get_velocity()
            kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
            # if kmh < speedMinimum:
            #     maxSteer = 0.01
            # elif kmh < speedTarget:
            #     maxSteer = 0.1
            # else:
            #     maxSteer = 1
            if abs(deltaY) < thresholdDeltaYSteer:
                # deltaY = -deltaY
                maxSteer = 1e-2
            else:
                maxSteer = 1e-1
            if deltaY >= -thresholdDeltaYNoSteer and deltaY <= thresholdDeltaYNoSteer:
                bWithinThreshold = True
                throttle, steer, brake = getStandardVehicleControl()
            elif deltaY > thresholdDeltaYNoSteer:
                bWithinThreshold = False
                deltaSteer = -unitChangeSteer
                steer = max(steer+deltaSteer, -maxSteer)
            elif deltaY < thresholdDeltaYNoSteer:
                bWithinThreshold = False
                deltaSteer = unitChangeSteer
                steer = min(steer+deltaSteer, maxSteer)
            if kmh < speedTarget:
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
            print(f'tick: {countTick} | distance to destination: {getDistanceToDestination():.1f} | deltaY: {deltaY:.2f} | throttle: {throttle:.2f} steer: {steer:.4f} brake: {brake:.1f}')
            world.tick()
            countTick += 1
        # Save the delta Y plot
        ax1.plot(listDeltaY)
        fig_deltaY.savefig(os.path.join(dir_outptut, 'deltaY.png'))
        plt.close(fig_deltaY)
        # Save the overlay plot
        x_vehicle = [location.x for location in listLocations]
        y_vehicle = [10*(location.y-location_destination.y) for location in listLocations]
        x_path = [location.x for location in listLocationsPath_CARLA_AP_Town06]
        y_path = [10*(location.y-location_destination.y) for location in listLocationsPath_CARLA_AP_Town06]
        ax2.plot(x_vehicle, y_vehicle, label='Vehicle Location', marker='o', linestyle='-')
        ax2.plot(x_path, y_path, label='Path Location', marker='o', linestyle='--')
        ax2.legend()
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Vehicle Location and Path Overlay')
        # Adjust the ylim values to zoom in on the y-values
        y_range = max(y_path) - min(y_path)
        # ax2.set_ylim(244.6 - 50, 244.6 + 50)
        fig_overlay.savefig(os.path.join(dir_outptut, 'overlay_plot.png'))
        plt.close(fig_overlay)

        time.sleep(10)
        while not os.path.join(dir_output_frames, f'{countTick:06d}.png'):
            time.sleep(10)
    finally:
        actor_list_destroy(actor_list)
        print('done')

if __name__ == '__main__':
    main()