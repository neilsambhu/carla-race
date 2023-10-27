import carla

def main():
	'''Make sure CARLA Simulator 0.9.14 is running'''
	
	# Connect to the CARLA Simulator
	client = carla.Client('localhost', 2000)
	client.set_timeout(20.0)

	# Get the world object
	world = client.get_world()
	world = client.load_world('Town04_Opt')

	# Define the blueprint of the vehicle you want to spawn
	blueprint_library = world.get_blueprint_library()
	vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

	# Now we need to give an initial transform to the vehicle. We choose a
	# random transform from the list of recommended spawn points of the map.
	print(world.get_map().get_spawn_points())
	# transform = random.choice(world.get_map().get_spawn_points())

if __name__ == '__main__':
    main()