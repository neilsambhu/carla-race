import carla, time, queue, shutil, os

def actor_list_destroy(actor_list):
    [x.destroy() for x in actor_list]
    return []
    
def main():
    '''Make sure CARLA Simulator 0.9.14 is running'''
    actor_list = []

    if os.path.exists('_out_11train_rl'):
        shutil.rmtree('_out_11train_rl')
    
    try:
        # Connect to the CARLA Simulator
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)

        # Get the world object
        world = client.get_world()
        world = client.load_world('Town04_Opt')

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
        transform = world.get_map().get_spawn_points()[0]

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

        # # delay for vehicle to spawn
        # for i in range(10):
        #     world.tick()
        #     time.sleep(1)
        # time.sleep(5)
        
        # Now we register the function that will be called each time the sensor
        # receives an image. In this example we are saving the image to disk.
        # camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame, cc))
        camera.listen(lambda image: image.save_to_disk('_out_11train_rl/%06d.png' % image.frame))

        x_groundTruth,y_groundTruth,z_groundTruth=0.0,0.0,0.0
        x_train,y_train,z_train=0.0,0.0,0.0
        with open('_out_07vehicle_location_AP/Town04_0_335.txt','r') as file:
            for line in file.readlines():
                world.tick()
                lineStripped = line.strip()
                x_groundTruth,y_groundTruth,z_groundTruth = lineStripped.split()
                location_groundTruth = carla.Location(float(x_groundTruth),float(y_groundTruth),float(z_groundTruth))
                location_train = vehicle.get_location()
                # print(f'location_groundTruth: {location_groundTruth}\tlocation_train: {location_train}')
                distance = location_train.distance(location_groundTruth)
                print(f'distance: {distance}')
                

    finally:
        actor_list_destroy(actor_list)
        print('done')

if __name__ == '__main__':
    main()