import os
import numpy as np

def plot_circle(center, radius, resolution=100):
    theta = np.linspace(0, 2 * np.pi, resolution)  # 원을 그리기 위한 각도 범위
    x = center[0] + radius * np.cos(theta)  # x 좌표 계산
    z = center[2] + radius * np.sin(theta)  # z 좌표 계산
    y = np.full_like(x, center[1])  # y 좌표는 center[1] 값으로 동일
    return x, y, z

center = (0, 0, 0)  # 원의 중심 좌표
radius = 1.0  # 원의 반지름

x, y, z = plot_circle(center, radius)

# 결과 출력
for i in range(1,10):
    os.system('python ray_ver1.1.py {1} {0},{2},0.0'.format(x[i],i,z[i]))