from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

height = 100
width = 100

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# 각 프로세스가 처리할 영역 계산
N = height // size + (height % size > rank)
end = comm.scan(N)
start = end - N
print(("{3}:{0},{2},{1}").format(N,end,start,rank))
# 각 프로세스가 생성한 부분 이미지 배열
local_image = np.zeros((N, width, 3,))

# 부분 이미지 생성
for y in range(N):
    for x in range(width):
        local_image[y, x, 0] = (x / width) # R 채널에 x 좌표에 비례한 값을 대입
        local_image[y, x, 1] = (y / N) # G 채널에 y 좌표에 비례한 값을 대입
        local_image[y, x, 2] = 0  # B 채널에 0을 대입 (파란색은 0)
# 모든 프로세스에서 생성된 부분 이미지를 수집하여 전체 이미지 생성
recvbuf = None
if rank == 0:
    recvbuf = np.empty((height, width, 3))

# 부분 이미지를 전체 이미지로 모으기
comm.Gatherv(local_image, recvbuf, root=0)

# 0번 프로세스에서만 이미지 저장
if rank == 0:
    filename = "testing_01.png"
    plt.imsave(filename, recvbuf)
    print(f"이미지가 {filename}으로 저장되었습니다.")

