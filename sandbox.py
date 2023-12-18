import carla

def spawn_vehicle(client):
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Get a vehicle blueprint (e.g., a Tesla Model 3)
    vehicle_bp = blueprint_library.filter('model3')[0]

    # Get a random spawn point in the world
    spawn_point = world.get_random_location_from_navigation()

    # Create a transform from the spawn point
    transform = carla.Transform(spawn_point)

    # Spawn the vehicle at the selected spawn point
    vehicle = world.spawn_actor(vehicle_bp, transform)

    return vehicle

# Connect to CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(60.0)  # seconds

# Spawn a vehicle
spawned_vehicle = spawn_vehicle(client)

# Do something with the spawned vehicle...
# For example, print the vehicle ID
if spawned_vehicle:
    print(f"Spawned vehicle {spawned_vehicle}")
    spawned_vehicle.destroy()
    import time; time.sleep(1)
    print(f"Spawned vehicle {spawned_vehicle}")
