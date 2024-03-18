import carla, time, queue, shutil, os

def main():
    '''Make sure CARLA Simulator 0.9.14 is running'''
    
    try:
        # Connect to the CARLA Simulator
        # client = carla.Client('10.247.52.30', 2000)
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0) # already tuned

        # Get the world object
        world = client.get_world()
        world = client.load_world('Town06_Opt')

        # # Set synchronous mode
        # settings = world.get_settings()
        # settings.synchronous_mode = True # Enables synchronous mode
        # # settings.fixed_delta_seconds = 0.05
        # settings.fixed_delta_seconds = 0.1
        # # settings.fixed_delta_seconds = 0.01
        # # settings.fixed_delta_seconds = 0.001
        # world.apply_settings(settings)

    finally:
        pass

if __name__ == '__main__':
    main()