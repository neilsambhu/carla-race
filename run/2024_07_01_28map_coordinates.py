import carla
import math
import numpy as np
import matplotlib.pyplot as plt

def calculate_smoothed_yaw(locations, idx, window_size=5):
    # Calculate the average yaw over a sliding window
    start_idx = max(0, idx - window_size)
    end_idx = min(len(locations) - 1, idx + window_size)
    yaw_sum = 0
    count = 0
    
    for i in range(start_idx, end_idx):
        if i == end_idx - 1:
            continue
        yaw_sum += calculate_orientation(locations[i], locations[i + 1])
        count += 1
    
    return yaw_sum / count if count > 0 else 0

def calculate_orientation(prev_loc, next_loc):
    # Calculate the yaw (orientation) between two points
    dx = next_loc.x - prev_loc.x
    dy = next_loc.y - prev_loc.y
    yaw = math.degrees(math.atan2(dy, dx))
    return yaw

def get_road_boundaries(locations, lane_width=3.5, smoothing_window=5):
    road_boundaries = []

    for i in range(1, len(locations) - 1):
        curr_location = locations[i]

        # Calculate smoothed orientation (yaw) based on a sliding window
        yaw = calculate_smoothed_yaw(locations, i, smoothing_window)

        # Calculate the left and right boundary locations
        left_boundary = carla.Location(
            curr_location.x - lane_width * math.sin(math.radians(yaw)) * 0.5,
            curr_location.y + lane_width * math.cos(math.radians(yaw)) * 0.5,
            curr_location.z
        )

        right_boundary = carla.Location(
            curr_location.x + lane_width * math.sin(math.radians(yaw)) * 0.5,
            curr_location.y - lane_width * math.cos(math.radians(yaw)) * 0.5,
            curr_location.z
        )

        road_boundaries.append((left_boundary, right_boundary))

    return road_boundaries
def getRoadBoundaries(locations):
    for location in locations:
        print(location)

def plot_map_boundaries(road_boundaries):
    left_x = [left.x for left, _ in road_boundaries]
    left_y = [left.y for left, _ in road_boundaries]
    right_x = [right.x for _, right in road_boundaries]
    right_y = [right.y for _, right in road_boundaries]

    plt.figure(figsize=(10, 10))
    plt.plot(left_x, left_y, 'r-', label='Left Boundary')
    plt.plot(right_x, right_y, 'b-', label='Right Boundary')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Road Boundaries from Ground-Truth Drive')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('road_boundaries_from_drive.png')
    plt.show()

def main():
    strPathType = 'Loop'
    path_AP_locations = f'_out_21_CARLA_AP_Town06/Locations{strPathType}.txt'
    def getPath_CARLA_AP_Town06():
        listLocationsPath_CARLA_AP_Town06 = []
        with open(path_AP_locations,'r') as file_AP_locations:
            for line in file_AP_locations.readlines():
                lineStripped = line.strip()
                x,y,z = lineStripped.split()
                locationFromPath = carla.Location(float(x),float(y),float(z))
                listLocationsPath_CARLA_AP_Town06.append(locationFromPath)
        return listLocationsPath_CARLA_AP_Town06
    listLocationsPath_CARLA_AP_Town06 = getPath_CARLA_AP_Town06()
    # Assume you have a list of 3378 ground-truth locations
    locations = listLocationsPath_CARLA_AP_Town06  # This should be a list of carla.Location objects

    # Convert the list of locations into road boundaries
    road_boundaries = get_road_boundaries(locations)
    road_boundaries=road_boundaries[3376//4*2+325:-1325]
    locations=locations[3376//4*2+325:-1325]
    print(len(road_boundaries))
    # listRoadBoundaries=None
    # getRoadBoundaries(locations);quit()

    # Print the road boundaries in CARLA coordinates on one line
    # for left, right in road_boundaries:
    for location,(left,right) in zip(locations,road_boundaries):
        sOut=''
        sOut+=f'{location} | '
        sOut+=f'L Bound: x={left.x:06.1f}'
        sOut+=f', y={left.y:06.1f}, z={left.z:06.1f} | '
        sOut+=f'R Bound: x={right.x:06.1f}, '
        sOut+=f'y={right.y:06.1f}, z={right.z:06.1f}'
        print(sOut)

    # Plot and save the road boundaries
    plot_map_boundaries(road_boundaries)

if __name__ == '__main__':
    main()
