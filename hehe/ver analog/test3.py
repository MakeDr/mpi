from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

start_time = MPI.Wtime()

max_depth = 3

width = 1024
height = 720

camera = np.array([0, 0, 1])
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

print("rank{}:{},{},{},{}".format(rank,N,R_N,start_row,end_row))

image = np.zeros((end_row-start_row, width, 3))
Y = np.linspace(screen[1], screen[3], height)[start_row:end_row]
X = np.linspace(screen[0], screen[2], width)
for i, y in enumerate(Y):
    for j, x in enumerate(X):
        image[i, j, 0] = x / (width - 1)  # R 채널에 x 좌표에 비례한 값을 대입 (0에서 1 사이로 정규화)
        image[i, j, 1] = y / (end_row-start_row - 1)  # G 채널에 y 좌표에 비례한 값을 대입 (0에서 1 사이로 정규화)
        image[i, j, 2] = 0  # B 채널에 0을 대입 (파란색은 0)

# 이미지 수집
final_image = None
if rank == 0:
    final_image = np.empty((height, width, 3))

comm.Gather(image, final_image, root=0)

# 랭크 0 프로세스에서 결과 이미지를 저장
if rank == 0:
    final_image = np.clip(final_image, 0, 1)  # 이미지 값을 0에서 1 사이로 클리핑
    plt.imsave('image4.png', final_image)
    end_time = MPI.Wtime()
    print("Overall elapsed time: " + str(end_time-start_time))