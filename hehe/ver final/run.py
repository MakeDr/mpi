import os
import numpy as np

def plot_circle(center, radius, resolution=100):
    theta = np.linspace(0, 2 * np.pi, resolution)  # 원을 그리기 위한 각도 범위
    x = center[0] + radius * np.cos(theta)  # x 좌표 계산
    y = center[2] + radius * np.sin(theta)  # z 좌표 계산
    return x, y

center = (0, 0, 0)  # 원의 중심 좌표
radius = 0.5  # 원의 반지름

x, y = plot_circle(center, radius)

# 결과 출력
for i in range(0,100):
    os.system('python ray_new.py {0} {1},{2},1.0'.format(i,x[i],y[i]))