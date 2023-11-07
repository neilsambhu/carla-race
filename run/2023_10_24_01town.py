import carla

def main():
	'''Make sure CARLA Simulator 0.9.14 is running'''
	
	# Connect to the CARLA Simulator
	client = carla.Client('localhost', 2000)
	client.set_timeout(20.0)

	# Get the world object
	world = client.get_world()
	world = client.load_world('Town04_Opt')

	# Set synchronous mode
	settings = world.get_settings()
	settings.synchronous_mode = True # Enables synchronous mode
	# settings.fixed_delta_seconds = 0.05
	settings.fixed_delta_seconds = 0.1
	world.apply_settings(settings)

	# client.reload_world(False) # reload map keeping the world settings
	# client.reload_world() # reload map keeping the world settings

	# # Simulation loop
	# while True:
	# 	# Your code
	# 	world.tick()
	world.tick()

	# print(carla.Landmark)

if __name__ == '__main__':
    main()