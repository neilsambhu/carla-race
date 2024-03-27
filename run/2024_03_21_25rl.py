import carla, time, queue, shutil, os, glob, math, configparser, subprocess, cv2
import numpy as np
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('config.ini')
bSAMBHU23 = config.getboolean('Settings','bSAMBHU23')
bGAIVI = not bSAMBHU23

# strPathType = 'Straight'
# strPathType = 'Curve'
strPathType = 'Loop'
path_AP_controls = f'_out_21_CARLA_AP_Town06/Controls{strPathType}.txt'
path_AP_locations = f'_out_21_CARLA_AP_Town06/Locations{strPathType}.txt'

dir_outptut = '_out_25_rl'
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
def strPoint(point):
    return f'{point:05.1f}'
def strLocation2D(location):
    return f'{strPoint(location.x)}, {strPoint(location.y)}'
def strLocation3D(location):
    return f'{strPoint(location.x)}, {strPoint(location.y)}, {strPoint(location.z)}'
def Point_ToString(point):
    return f'{point:06.2f}'
def Vector3D_ToString(vector3D):
    return f'{Point_ToString(vector3D.x)}, {Point_ToString(vector3D.y)}, {Point_ToString(vector3D.z)}'
def str_kmh(kmh):
    return f'{kmh:05.1f}'
def VehicleSpeed1D(vehicle):
    v = vehicle.get_velocity()
    kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
    return kmh
def Distance(seconds, velocity, acceleration):
    return velocity*seconds + 0.5*acceleration*seconds**2
def Magnitude3Dto1D(v):
    # print(f'v: {v}')
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)
def Location250msPrediction(fps, countTick, vehicle):
    # predict 5 frames away at 20 FPS
    # deltaT = 0.05
    deltaT = 0.25
    # deltaT = 5
    output = ''
    output += f'cur loc: {Vector3D_ToString(vehicle.get_location())} | '
    # output += f'cur acc: {Vector3D_ToString(vehicle.get_acceleration())} | '
    distance = Distance(deltaT, vehicle.get_velocity(), vehicle.get_acceleration())
    # output += f'dist: {Vector3D_ToString(distance)} | '
    locationPrediction = vehicle.get_location()+distance
    tickPrediction = int(countTick + fps*deltaT)
    output += f'pred loc at tick {tickPrediction:04d}: {Vector3D_ToString(locationPrediction)} | '
    return locationPrediction, tickPrediction, output
def Z_VelocitySmall(vehicle):
    zVelocityThreshold = 0.01
    return abs(vehicle.get_velocity().z)<zVelocityThreshold
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
        # settings.fixed_delta_seconds = 0.01
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
        spawn_point = carla.Transform(
                carla.Location(x=-313.8, y=243.6, z=0.1),
                carla.Rotation()
            )
        location_destination_straight = carla.Location(x=581.2, y=244.6, z=height)
        location_destination_curve = carla.Location(x=664.9, y=168.2, z=height)
        transform = spawn_point
        location_destination = spawn_point.location

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
        # vehicle.set_simulate_physics(False)

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
        # listDeltaY = []
        listDeltaTheta = []
        listLocations = []
        # Plot setup for delta Y
        # fig_deltaY, ax1 = plt.subplots(figsize=(12, 6))
        fig_deltaTheta, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('Time Steps')
        # ax1.set_ylabel('Delta Y')
        ax1.set_ylabel('Delta Theta')
        # ax1.set_title('Delta Y over Time')
        ax1.set_title('Delta Theta over Time')
        # Plot setup for overlay
        fig_overlay, ax2 = plt.subplots(figsize=(12, 6))  # Adjust the figsize as needed
        # leg = ax2.legend()
        # for line in leg.get_lines():
        #     line.set_linewidth(1)
        # fig_overlay, ax2 = plt.subplots(figsize=(12, 12))  # Adjust the figsize as needed
        list_x = [location.x for location in listLocationsPath_CARLA_AP_Town06]
        list_y = [location.y for location in listLocationsPath_CARLA_AP_Town06]
        left = min(list_x)
        bottom = min(list_y)
        top = max(list_y)
        width = max(list_x) - min(list_x)
        height = max(list_y) - min(list_y)
        # print(left, bottom, width, height)
        # fig_overlay, ax2 = plt.axes([left, bottom, width, height])
        # fig_overlay.add_axes(plt.axes([left, bottom, width, height]))
        # ax2.set_yticks(np.arange(bottom, top, 1))
        # ax2.set_aspect('equal', 'box')
        ax2.set_aspect('auto', 'box')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Vehicle Location and Path Overlay')
        def printLocations(currentLocation, closestLocation):
            return f'current location: {strLocation2D(currentLocation)} | closest location from path: {strLocation2D(closestLocation)}'
        dictLocationPrediction = {}
        def GetVehicleControls_legacy(throttle, steer, brake, locationPrediction, locationClosestToPredicted):
            deltaY = locationPrediction.y - locationClosestToPredicted.y
            listDeltaY.append(deltaY)
            listLocations.append(vehicle.get_location())
            thresholdDeltaYNoSteer = 0.5e-10
            thresholdDeltaYSteer = 1e-1
            speedMinimum = 10
            speedTarget = 15
            bWithinThreshold = None
            maxSteer = None
            unitChangeThrottle = 0.1
            unitChangeSteer = 1
            unitChangeBrake = 0.1
            kmh = VehicleSpeed1D(vehicle)
            # output += f'{str_kmh(kmh)} | '
            if kmh < speedMinimum:
                maxSteer = 0.01
            else:
                maxSteer = min(abs(deltaY)/10, 0.01)
            # if abs(deltaY) < thresholdDeltaYSteer:
            #     # deltaY = -deltaY
            #     maxSteer = 1e-3
            # else:
            #     maxSteer = 1e-1
            if deltaY >= -thresholdDeltaYNoSteer and deltaY <= thresholdDeltaYNoSteer:
                bWithinThreshold = True
                throttle, steer, brake = getStandardVehicleControl()
            elif deltaY > thresholdDeltaYNoSteer:
                bWithinThreshold = False
                deltaSteer = -unitChangeSteer
                steer = max(steer+deltaSteer, -maxSteer)
            elif deltaY < -thresholdDeltaYNoSteer:
                bWithinThreshold = False
                deltaSteer = unitChangeSteer
                steer = min(steer+deltaSteer, maxSteer)
            if not bWithinThreshold:
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
            return throttle, steer, brake
        def unit_vector(vector):
            """ Returns the unit vector of the vector.  """
            return vector / np.linalg.norm(vector)
        def angle_between_legacy(v1, v2):
            """ Returns the angle in radians between vectors 'v1' and 'v2'::

                    >>> angle_between((1, 0, 0), (0, 1, 0))
                    1.5707963267948966
                    >>> angle_between((1, 0, 0), (1, 0, 0))
                    0.0
                    >>> angle_between((1, 0, 0), (-1, 0, 0))
                    3.141592653589793
            """
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        def angle_between(vector1, vector2):
            dotProduct = float(vector1[0]*vector2[0] + vector1[1]*vector2[1])
            magnitude = float((vector1[0]**2 + vector1[1]**2)**(1/2) * (vector2[0]**2 + vector2[1]**2)**(1/2))
            # return np.arccos(dotProduct/magnitude)
            import math
            # print(dotProduct/magnitude)
            division = None
            if magnitude == 0:
                division = 1
            else:
                division = dotProduct/magnitude
            # division = min(dotProduct/magnitude, 1.0)
            return math.acos(division)
        def GetVehicleOutput(theta, locationClosestToPredicted):
            # x = vehicle.get_location().x*math.cos(theta) - vehicle.get_location().y*math.sin(theta)
            # y = vehicle.get_location().x*math.sin(theta) + vehicle.get_location().y*math.cos(theta)
            x = locationClosestToPredicted.x*math.cos(theta) - locationClosestToPredicted.y*math.sin(theta)
            y = locationClosestToPredicted.x*math.sin(theta) + locationClosestToPredicted.y*math.cos(theta)
            return x, y
        def GetTurnDirection(a, b, c):
            output = (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x)
            if output < 0:
                return -1
            if output == 0:
                return 0
            if output > 0:
                return 1
        def GetVehicleControls(throttle, steer, brake, locationPrediction, locationClosestToPredicted):
            output = ''
            # npLocationCurrent = np.array([vehicle.get_location().x, vehicle.get_location().y, vehicle.get_location().z])
            npLocationCurrent = np.array([vehicle.get_location().x, vehicle.get_location().y])
            # npLocationPrediction = np.array([locationPrediction.x, locationPrediction.y, locationPrediction.z])
            npLocationPrediction = np.array([locationPrediction.x, locationPrediction.y])
            # npLocationClosestToPredicted = np.array([locationClosestToPredicted.x, locationClosestToPredicted.y, locationClosestToPredicted.z])
            npLocationClosestToPredicted = np.array([locationClosestToPredicted.x, locationClosestToPredicted.y])
            # vector_currToPred = npLocationCurrent - npLocationPrediction
            # vector_currToPred = 0.01*(-npLocationCurrent + npLocationPrediction)
            vector_currToPred = -npLocationCurrent + npLocationPrediction
            # vector_currToClosestToPredicted = npLocationCurrent - npLocationClosestToPredicted
            # vector_currToClosestToPredicted = 0.01*(-npLocationCurrent + npLocationClosestToPredicted)
            vector_currToClosestToPredicted = -npLocationCurrent + npLocationClosestToPredicted
            # output += f'vector_currToPred: {vector_currToPred} | vector_currToClosestToPredicted: {vector_currToClosestToPredicted} | '
            turnDirection = GetTurnDirection(vehicle.get_location(), locationPrediction, locationClosestToPredicted)
            output += f'turnDirection: {turnDirection} | '
            # multiply by -1 to account for left is negative and right is positive, not like unit circle
            deltaTheta = 0.2*-1*turnDirection*angle_between(vector_currToPred, vector_currToClosestToPredicted)
            # deltaTheta = -1*turnDirection*angle_between(vector_currToPred, vector_currToClosestToPredicted)
            deltaTheta = math.degrees(deltaTheta)
            output += f'theta {deltaTheta:.1f} | '
            # output = f'theta {deltaTheta:.2f} | '
            # x, y = GetVehicleOutput(deltaTheta, locationClosestToPredicted)
            # # output = f'x, y: {x:.1f}, {y:.1f}'
            listDeltaTheta.append(deltaTheta)
            listLocations.append(vehicle.get_location())
            thresholdDeltaThetaNoSteer = 0.5e-10
            # thresholdDeltaThetaNoSteer = 5
            thresholdDeltaThetaSteer = 1e-1
            speedMinimum = 5
            speedTarget = 30
            bWithinThreshold = None
            maxSteer = None
            unitChangeThrottle = 0.1
            unitChangeSteer = 0.1
            unitChangeBrake = 0.1
            kmh = VehicleSpeed1D(vehicle)
            # output += f'{str_kmh(kmh)} | '
            if kmh < speedMinimum:
                maxSteer = 0.01
            else:
                # maxSteer = min(abs(deltaTheta)/10, 0.01)
                # maxSteer = min(abs(deltaTheta)/10, 0.3)
                maxSteer = min(abs(deltaTheta)/10, 1)
            # if abs(deltaTheta) < thresholdDeltaThetaSteer:
            #     # deltaTheta = -deltaTheta
            #     maxSteer = 1e-3
            # else:
            #     maxSteer = 1e-1
            if deltaTheta >= -thresholdDeltaThetaNoSteer and deltaTheta <= thresholdDeltaThetaNoSteer:
                bWithinThreshold = True
                throttle, steer, brake = getStandardVehicleControl()
            elif deltaTheta > thresholdDeltaThetaNoSteer:
                bWithinThreshold = False
                deltaSteer = -unitChangeSteer
                steer = max(steer+deltaSteer, -maxSteer)
            elif deltaTheta < -thresholdDeltaThetaNoSteer:
                bWithinThreshold = False
                deltaSteer = unitChangeSteer
                steer = min(steer+deltaSteer, maxSteer)
            if not bWithinThreshold:
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
            return throttle, steer, brake, output
        while getDistanceToDestination() > 2 or countTick < 500:
            output = f'tick: {countTick:04d} | '
            if not Z_VelocitySmall(vehicle):
                print(output)
                world.tick()
                countTick += 1
                continue
            locationPrediction, tickPrediction, output_temp = Location250msPrediction(1/settings.fixed_delta_seconds, countTick, vehicle)
            dictLocationPrediction[tickPrediction] = locationPrediction
            output += output_temp            
            if countTick in dictLocationPrediction:
                distanceError = abs(vehicle.get_location()-dictLocationPrediction[countTick])
                # output += f'pred err: {Vector3D_ToString(distanceError)} | '
            locationClosestToPredicted = getLocationClosestToCurrent(locationPrediction)
            output += f'loc closest to pred: {Vector3D_ToString(locationClosestToPredicted)} | '
            distancePredictionAndPath = locationPrediction.distance(locationClosestToPredicted)
            output += f'pred->path dist: {distancePredictionAndPath:.2f} | '
            throttle, steer, brake, output_temp = GetVehicleControls(throttle, steer, brake, locationPrediction, locationClosestToPredicted)
            output += output_temp
            vehicleControl = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
            vehicle.apply_control(vehicleControl)
            print(output)
            world.tick()
            countTick += 1
        # Save the delta Y plot
        # ax1.plot(listDeltaY)
        ax1.plot(listDeltaTheta)
        # fig_deltaY.savefig(os.path.join(dir_outptut, 'deltaY.png'))
        fig_deltaTheta.savefig(os.path.join(dir_outptut, 'deltaTheta.png'))
        # plt.close(fig_deltaY)
        plt.close(fig_deltaTheta)
        # Save the overlay plot
        # stretch = 100
        stretch = 1
        x_vehicle = [location.x for location in listLocations]
        y_vehicle = [location.y for location in listLocations]
        # y_vehicle = [stretch*(location.y-location_destination.y) for location in listLocations]
        x_path = [location.x for location in listLocationsPath_CARLA_AP_Town06]
        y_path = [location.y for location in listLocationsPath_CARLA_AP_Town06]
        # y_path = [stretch*(location.y-location_destination.y) for location in listLocationsPath_CARLA_AP_Town06]
        ax2.plot(x_vehicle, y_vehicle, label='Vehicle Location', marker='o', linestyle='-', linewidth=0.01)
        ax2.plot(x_path, y_path, label='Path Location', marker='o', linestyle='--', linewidth=0.01)
        ax2.legend()
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Vehicle Location and Path Overlay')
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