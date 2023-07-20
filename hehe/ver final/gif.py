import imageio
import numpy as np

# 0부터 99까지의 배열 생성
numbers = np.arange(100)
image_files = ["image_{}.png".format(i) for i in numbers]
print(image_files)
# 이미지 로드
images = [imageio.imread(image_file) for image_file in image_files]

# 이미지를 결합하여 GIF 생성
imageio.mimsave('animated.gif', images, duration=0.02)  # 모든 이미지를 결합해 animated.gif로 저장합니다. 여기서 duration은 각 프레임 간의 시간 간격(초)입니다. 값을 조정하면 프레임 전환 시간을 조절할 수 있습니다.

print("GIF 생성 완료!")
