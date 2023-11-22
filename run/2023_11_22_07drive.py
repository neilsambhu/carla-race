import carla, time, queue, shutil, os

def main():
    '''Make sure CARLA Simulator 0.9.14 is running'''
    actor_list = []

    if os.path.exists('_out_04drive'):
        shutil.rmtree('_out_04drive')
    
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
        transform = world.get_map().get_spawn_points()[3]

        # So let's tell the world to spawn the vehicle.
        vehicle = world.spawn_actor(vehicle_bp, transform)

        # It is important to note that the actors we create won't be destroyed
        # unless we call their "destroy" function. If we fail to call "destroy"
        # they will stay in the simulation even after we quit the Python script.
        # For that reason, we are storing all the actors we create so we can
        # destroy them afterwards.
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(False)

        # Let's add now a "depth" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)

        # # delay for vehicle to spawn
        # for i in range(10):
        #     world.tick()
        #     time.sleep(1)
        # time.sleep(5)
        
        # Now we register the function that will be called each time the sensor
        # receives an image. In this example we are saving the image to disk.
        # camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame, cc))
        camera.listen(lambda image: image.save_to_disk('_out_04drive/%06d.png' % image.frame))

        throttle,steer,brake,hand_brake,reverse,manual_gear_shift,gear=0.5,0.0,0.0,False,False,False,0
        with open('_out_control/control.txt','r') as file:
            for line in file.readlines():
                lineStripped = line.strip()
                # print(lineStripped.split())
                throttle,steer,brake,hand_brake,reverse,manual_gear_shift,gear = lineStripped.split()
                control = carla.VehicleControl(
                    throttle=float(throttle),
                    steer=float(steer),
                    brake=float(brake),
                    hand_brake=hand_brake=='True',
                    reverse=reverse=='True',
                    manual_gear_shift=manual_gear_shift=='True',
                    gear=int(gear))
                vehicle.apply_control(control)
                world.tick()
                time.sleep(1)

    finally:
        print('destroying actors')
        camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')

if __name__ == '__main__':
    main()