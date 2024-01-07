import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Path to Town04.jpg
map_img_path = 'maps/Town04.jpg'

# Load Town04.jpg as the background map
map_img = mpimg.imread(map_img_path)

# Get image dimensions
image_height, image_width, _ = map_img.shape

# Path to the locations file
locations_file_path = '_out_07CARLA_AP/Locations_Town04_0_335.txt'

# Read the locations file and extract x, y coordinates
with open(locations_file_path, 'r') as file:
    # Read lines and split by spaces to get x, y, z coordinates
    lines = file.readlines()
    # lines = file.readlines()[0:3400]
    trajectory = []
    for line in lines:
        coordinates = line.strip().split(' ')
        # Consider only x and y coordinates, assuming they are in the first two columns
        x = float(coordinates[0])
        y = float(coordinates[1])
        trajectory.append((x, y))

# Separate x and y coordinates for plotting
x_coords, y_coords = zip(*trajectory)

# Shift the coordinates to set the origin at the center of the image
x_coords_shifted = [x - image_width / 2 for x in x_coords]
y_coords_shifted = [image_height / 2 - y for y in y_coords]

# Plot the map
plt.imshow(map_img)

# Plot the shifted vehicle's path
plt.plot(x_coords, y_coords, marker='o', color='red', markersize=2)  # Adjust marker size as needed
# plt.plot(x_coords_shifted, y_coords_shifted, marker='o', color='red', markersize=2)  # Adjust marker size as needed

# Show the plot
plt.show()
