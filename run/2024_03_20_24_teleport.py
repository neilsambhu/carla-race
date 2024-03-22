import carla, time
client = carla.Client('localhost', 2000)
world = client.get_world()
print(world)
a = world.get_actors()
time.sleep(1)
vehicle = world.get_actors().filter('vehicle.*')[0]
LOCATION = carla.Location(x=-313.8, y=243.6, z=0.1)
# LOCATION = carla.Location(x=-0.8, y=243.6, z=0.1)
transform = carla.Transform(LOCATION, carla.Rotation())
vehicle.set_transform(transform)