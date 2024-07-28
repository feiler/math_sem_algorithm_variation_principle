import numpy as np

def generate_unique_points(num_points, max_coord):
    points = set()
    while len(points) < num_points:
        point = tuple(np.random.randint(max_coord, size=2))
        points.add(point)
    return list(points)

# Generate unique random points
num_points = 30
max_coord = 100
points = generate_unique_points(num_points, max_coord)

print(points)
