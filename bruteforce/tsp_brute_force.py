import matplotlib.pyplot as plt
import numpy as np
import time
import itertools

# Define your cities here. For example:
#cities = np.array([(0, 16), (6, 56), (59, 77)])
#cities = np.array([(0, 16), (6, 56), (59, 77),(67, 35),(5, 13), (25, 16),(78, 4), (50, 81), (93, 43), (88, 61), (0, 32), (59, 63)])
cities = np.array([(0, 16), (6, 56), (59, 77), (67, 35), (5, 13)])

num_cities = len(cities)

# Function to calculate distance
def calculate_distance(city1, city2):
    return np.sqrt((city1[0]-city2[0])**2 + (city1[1]-city2[1])**2)

# Function to calculate total distance of a path
def total_distance(path):
    return sum(calculate_distance(cities[path[i-1]], cities[path[i]]) for i in range(len(path)))

# Initial path
path = list(range(num_cities))
best_path = path.copy()
best_distance = total_distance(path)

plt.ion() # Turn on interactive mode

# Generate all possible paths
paths = list(itertools.permutations(range(num_cities)))

start_time = time.time() # Start timer

for i, path in enumerate(paths): # Iterate over all paths
    # If the new path is shorter, update the best path and distance
    if total_distance(path) < best_distance:
        best_path = path
        best_distance = total_distance(path)
    
    # Plot cities
    plt.scatter(cities[:,0], cities[:,1])
    
    # Plot path
    for m in range(-1, len(path)-1):
        plt.plot((cities[path[m],0], cities[path[m+1],0]), (cities[path[m],1], cities[path[m+1],1]), 'r-')
    
    plt.title(f'Iteration {i+1}, Distance: {total_distance(path)}')
    plt.draw()
    plt.pause(0.1) # Pause for a while
    plt.clf() # Clear the figure

end_time = time.time() # End timer

plt.ioff() # Turn off interactive mode

# Plot the best path
plt.scatter(cities[:,0], cities[:,1])
for i in range(-1, len(best_path)-1):
    plt.plot((cities[best_path[i],0], cities[best_path[i+1],0]), (cities[best_path[i],1], cities[best_path[i+1],1]), 'r-')
plt.title(f'Best path with distance: {round(best_distance, 5)}, Total time: {round(end_time - start_time, 5)} seconds')
plt.show()
