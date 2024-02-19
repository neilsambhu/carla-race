import carla, time, queue, shutil, os, glob

def actor_list_destroy(actor_list):
    [x.destroy() for x in actor_list]
    return []
    
def main():
    '''Make sure CARLA Simulator 0.9.14 is running'''
    actor_list = []
    path_AP_controls = '_out_21_CARLA_AP_Town06/Controls.txt'
    path_AP_locations = '_out_21_CARLA_AP_Town06/Locations.txt'

    dir_outptut = '_out_22_rl'
    if os.path.exists(dir_outptut):
        shutil.rmtree(dir_outptut)
    os.makedirs(dir_outptut)
    dir_output_frames = f'{dir_outptut}/frames/'
    path_rl_controls = f'{dir_outptut}/Controls.txt'
    path_rl_locations = f'{dir_outptut}/Locations.txt'

    try:
        # Connect to the CARLA Simulator
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)

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
        spawn_destination = carla.Transform(carla.Location(x=581.2, y=244.6, z=height), carla.Rotation())
        transform = spawn_start_center

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

        throttle,steer,brake,hand_brake,reverse,manual_gear_shift,gear=0.5,0.0,0.0,False,False,False,0
        countTick = 0
        with open(path_AP_controls,'r') as file_AP_controls:
            file_rl_locations = open(path_rl_locations,'w')
            for line in file_AP_controls.readlines():
                lineStripped = line.strip()
                throttle,steer,brake = lineStripped.split()
                control = carla.VehicleControl(throttle=float(throttle),steer=float(steer),brake=float(brake))
                vehicle.apply_control(control)
                world.tick()
                countTick += 1
            countTick -= 1
        while len(glob.glob(os.path.join(dir_output_frames,'*'))) < countTick:
            time.sleep(1)
    finally:
        actor_list_destroy(actor_list)
        print('done')

if __name__ == '__main__':
    main()