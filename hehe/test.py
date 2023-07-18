import numpy as np
import matplotlib.pyplot as plt

height = 100
width = 200

image = np.zeros((height, width, 3), dtype=np.uint8)  # 이미지 배열을 정수 형식으로 생성 (0 ~ 255 범위)

for y in range(height):
    for x in range(width):
        image[y, x, 0] = int((x / width) * 255)  # R 채널에 x 좌표에 비례한 값을 대입
        image[y, x, 1] = int((y / height) * 255)  # G 채널에 y 좌표에 비례한 값을 대입
        image[y, x, 2] = 0  # B 채널에 0을 대입 (파란색은 0)

# 이미지 저장
filename = "testing_00.png"
plt.imsave(filename, image)

print(f"이미지가 {filename}으로 저장되었습니다.")

