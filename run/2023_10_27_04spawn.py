import carla, time, queue, shutil, os

def actor_list_destroy(actor_list):
    [x.destroy() for x in actor_list]
    return []

def main():
    '''Make sure CARLA Simulator 0.9.14 is running'''
    actor_list = []

    if os.path.exists('_out_04spawn'):
        shutil.rmtree('_out_04spawn')
    
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
        # print(f'len(world.get_map().get_spawn_points()): {len(world.get_map().get_spawn_points())}')
        for idx_spawn_point, spawn_point in enumerate(world.get_map().get_spawn_points()):
            # print(f'spawn_point.location: {spawn_point.location}\tspawn_point.rotation: {spawn_point.rotation}')
            transform = world.get_map().get_spawn_points()[idx_spawn_point]

            # So let's tell the world to spawn the vehicle.
            vehicle = world.spawn_actor(vehicle_bp, transform)

            # It is important to note that the actors we create won't be destroyed
            # unless we call their "destroy" function. If we fail to call "destroy"
            # they will stay in the simulation even after we quit the Python script.
            # For that reason, we are storing all the actors we create so we can
            # destroy them afterwards.
            actor_list.append(vehicle)
            # print('created %s' % vehicle.type_id)

            # Let's put the vehicle to drive around.
            vehicle.set_autopilot(True)

            # Let's add now a "depth" camera attached to the vehicle. Note that the
            # transform we give here is now relative to the vehicle.
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            actor_list.append(camera)
            # print('created %s' % camera.type_id)

            # Now we register the function that will be called each time the sensor
            # receives an image. In this example we are saving the image to disk.
            # camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame, cc))
            camera.listen(lambda image: image.save_to_disk('_out_04spawn/%03d_%06d.png' % (idx_spawn_point, image.frame)))

            world.tick()
            time.sleep(5e-1)
            actor_list = actor_list_destroy(actor_list)
            print(f'spawned {idx_spawn_point+1} of {len(world.get_map().get_spawn_points())}')

    finally:
        actor_list_destroy(actor_list)
        print('done')

if __name__ == '__main__':
    main()