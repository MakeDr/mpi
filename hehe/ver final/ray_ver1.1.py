from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance

def ray_tracing(x, y):
    pixel = np.array([x, y, 0]) 
    origin = camera 
    direction = normalize(pixel - origin) 
    color = np.zeros((3)) 
    reflection = 1 
    for k in range(max_depth): 
        nearest_object, min_distance = nearest_intersected_object(objects, origin, direction) 
        if nearest_object is None: 
            break 
        intersection = origin + min_distance * direction 
        normal_to_surface = normalize(intersection - nearest_object['center']) 
        shifted_point = intersection + 1e-5 * normal_to_surface 
        intersection_to_light = normalize(light['position'] - shifted_point) 
        _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light) 
        intersection_to_light_distance = np.linalg.norm(light['position'] - intersection) 
        is_shadowed = min_distance < intersection_to_light_distance 
        if is_shadowed: 
            break 
        illumination = np.zeros((3)) 
        illumination += nearest_object['ambient'] * light['ambient'] 
        illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface) 
        intersection_to_camera = normalize(camera - intersection) 
        H = normalize(intersection_to_light + intersection_to_camera) 
        illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4) 
        color += reflection * illumination 
        reflection *= nearest_object['reflection'] 
        origin = shifted_point 
        direction = reflected(direction, normal_to_surface)
    return color

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

start_time = MPI.Wtime()

max_depth = 10

import sys
width = 300
height = 200
# image count 읽어오기
if len(sys.argv) > 1:
    image_count = int(sys.argv[1])
else:
    image_count = 1  # 기본값 설정

# camera 위치 읽어오기
if len(sys.argv) > 2:
    camera_position = np.array([float(val) for val in sys.argv[2].split(",")])
else:
    camera_position = np.array([0, 0, 1])  # 기본값 설정
camera = camera_position
light = { 'position': np.array([5, 5, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }
objects = [
    { 'center': np.array([-0.2, 0, -1]), 'radius': 0.2, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 1, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 80, 'reflection': 0.1 },
    { 'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([0.5, 0, -1]), 'radius': 0.5, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0.7, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 }
]

ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom

N = height // size
R_N = height % size

if rank < R_N:
    start_row = rank * (N + 1)
    end_row = start_row + N + 1
else:
    start_row = rank * N + R_N
    end_row = start_row + N

image = np.zeros((end_row-start_row, width, 3))

Y = np.linspace(screen[1], screen[3], height)[start_row:end_row]
X = np.linspace(screen[0], screen[2], width)
for i, y in enumerate(Y):
    for j, x in enumerate(X):
        color = ray_tracing(x,y) 
        image[i, j] = (np.clip(color, 0, 1))*0.5 + 0.5

images = comm.gather(image, root=0)
if rank == 0:
    final_image = np.vstack(images)
    plt.imsave('image_{}.png'.format(image_count), final_image)
    end_time = MPI.Wtime()
    print("Overall elapsed time: " + str(end_time-start_time))