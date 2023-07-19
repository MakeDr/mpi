from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt


##---------------Fixed code---------------##
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
	# screen is on origin 
	pixel = np.array([x, y, 0]) 
	origin = camera 
	direction = normalize(pixel - origin) 
	color = np.zeros((3)) 
	reflection = 1 
	for k in range(max_depth): 
		# check for intersections 
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
		# ambiant 
		illumination += nearest_object['ambient'] * light['ambient'] 
		# diffuse 
		illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface) 
		# specular 
		intersection_to_camera = normalize(camera - intersection) 
		H = normalize(intersection_to_light + intersection_to_camera) 
		illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4) 
		# reflection 
		color += reflection * illumination 
		reflection *= nearest_object['reflection'] 
		origin = shifted_point 
		direction = reflected(direction, normal_to_surface)
	return color
##---------------Fixed code---------------##

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

start_time = MPI.Wtime()

max_depth = 3

#### parameters
width = 300
height = 200
camera = np.array([0, 0, 1])
#camera = np.array([0, 1, 1])
light = { 'position': np.array([5, 5, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }
objects = [
    { 'center': np.array([-0.2, 0, -1]), 'radius': 0.2, 'ambient': np.array([0, 1, 0]), 'diffuse': np.array([0.7, 1, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 80, 'reflection': 0.1 },
    { 'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([1, 0, 0]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([0.5, 0, -1]), 'radius': 0.5, 'ambient': np.array([1, 0, 1]), 'diffuse': np.array([0.7, 0.7, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0, 1]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 }
]

ratio = float(width) / height #비율

screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom

image = np.zeros((height, width, 3)) # 3차원 벡터에 0에 h 1에 w을 넣음 (높이,너비 순)


# 정적 분할 방식을 사용하여 각 프로세스에게 할당될 행(row) 범위 계산
N = height // size 
start_row = rank * comm.scan(N) 
end_row = start_row + N

print("{} : {},{},{}".format(rank,N,start_row,end_row))
# Y와 X 배열 생성
Y = np.linspace(screen[1], screen[3], height)
X = np.linspace(screen[0], screen[2], width)

# 레이 트레이싱을 수행하는 부분을 각 프로세스에게 할당된 행 범위로 제한
for i in range(start_row, end_row):
	y = Y[i]
	for j, x in enumerate(X):
		color = ray_tracing(x, y)
		image[i, j] = np.clip(color, 0, 1)

# 랭크 0 프로세스에서 결과 이미지를 수집할 버퍼 생성
result_image = None
if rank == 0:
    result_image = np.empty((height, width, 3))

# 랭크 0 프로세스에서 모든 부분 이미지 수집
comm.Gatherv(image[start_row:end_row], result_image, root=0)

# 랭크 0 프로세스에서 결과 이미지를 저장
if rank == 0:
    plt.imsave('image3.png', result_image)

end_time = MPI.Wtime()
if rank == 0:
    print("Overall elapsed time: " + str(end_time - start_time))
