### Referenced by https://ivo-lee.tistory.com/91 ###
# [install opencv]
# pip install opencv-python
# pip install opencv-contrib-python

### 활용 라이브러리 로드 ###
import cv2
# print(cv2.__version__) #OpenCV Version Check
import numpy as np
from matplotlib import pyplot as plt

### 흑백 이미지로 로드 ###
"""
image = cv2.imread("./Long_cat.jpg",cv2.IMREAD_GRAYSCALE) # 흑백 이미지로 로드
plt.imshow(image, cmap="gray"), plt.axis("off") # 이미지 출력
plt.show()
"""
### 이미지 데이터 확인 ###
"""
print("IMG TYPE :",type(image)) # 데이터 타입 확인
print("IMG ITSELF")
print(image)
print("IMG SHAPE :",image.shape)
"""
### 컬러 이미지로 로드 ###
"""
image_bgr = cv2.imread("./Long_cat.jpg",cv2.IMREAD_COLOR) # 흑백 이미지로 로드
print("IMG PIXEL :",image_bgr[0,0]) # 이미지 픽셀 확인
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # BGR이미지를 RGB로 변환
plt.imshow(image_rgb), plt.axis("off") # 이미지 출력
plt.show()
"""
### 이미지 저장 ###
"""
image = cv2.imread("./Long_cat.jpg",cv2.IMREAD_GRAYSCALE) # 흑백 이미지로 로드
cv2.imwrite("/new_img.jpg",image) # 이미지 저장
"""
### 이미지 크기 변경 ###
"""
# 머신러닝에서 많이 사용하는 이미지 크기 : 32x32, 64x64, 96x96, 256x256
image = cv2.imread("./Long_cat.jpg",cv2.IMREAD_GRAYSCALE) # 흑백 이미지로 로드
#image_50x50 = cv2.resize(image, (50,50)) # 이미지 크기를 50x50 픽셀로 변경
image_x256 = cv2.resize(image, (256,256)) # 이미지 크기를 256x256 픽셀로 변경
#plt.imshow(image_50x50, cmap="gray"), plt.axis("off") # 이미지를 출력
plt.imshow(image_x256, cmap="gray"), plt.axis("off") # 이미지를 출력
plt.show()
"""
### 이미지 자르기 ###
"""
image = cv2.imread("./Long_cat.jpg",cv2.IMREAD_GRAYSCALE) # 흑백 이미지로 로드
image_cropped = image[:,200:450] # 행렬값으로 이미지 분할, 원래는 이미지 크기를 맞춰서 작업하면 동일하게 자를 수 있음

plt.imshow(image_cropped, cmap="gray"), plt.axis('off') # 이미지를 출력
plt.show()
"""
### 이미지 투명도 처리 ###
"""
image = cv2.imread("./Long_cat.jpg",cv2.IMREAD_GRAYSCALE) # 흑백 이미지로 로드
# 각 픽셀 주변 5x5 커널 평균값으로 이미지 흐리게
image_blurry = cv2.blur(image, (5,5))

plt.imshow(image_blurry, cmap="gray"), plt.axis('off') # 이미지 출력
plt.show()
# 커널 크기의 영향을 강조하기 위해 50x50 커널로 같은 이미지를 흐리게
image_very_blurry = cv2.blur(image,(50,50))
plt.imshow(image_very_blurry, cmap="gray"), plt.xticks([]), plt.yticks([]) # 이미지를 출력
plt.show()
"""
### 이미지 투명도 알고리즘 (커널활용) ###
"""
image = cv2.imread("./Long_cat.jpg",cv2.IMREAD_GRAYSCALE) # 흑백 이미지로 로드
kernel = np.ones(((5,5))) / 25.0 # 커널 생성
print("KERNEL :",kernel)
image_kernel = cv2.filter2D(image, -1, kernel) # 커널을 적용
plt.imshow(image_kernel, cmap="gray"), plt.xticks([]), plt.yticks([]) # 이미지 출력
plt.show()

image_very_blurry = cv2.GaussianBlur(image, (5,5), 0) # 가우시안 블러를 적용
plt.imshow(image_very_blurry, cmap="gray"),plt.xticks([]), plt.yticks([]) # 이미지 출력
plt.show()

gaus_vector = cv2.getGaussianKernel(5, 0)
print("Gaus_Vector :",gaus_vector)
gaus_kernel = np.outer(gaus_vector,gaus_vector) # 벡터를 외적하여 커널을 생성
print("Gaus_Kernel :",gaus_kernel)

image_kernel = cv2.filter2D(image, -1, gaus_kernel) # 커널을 적용
plt.imshow(image_kernel, cmap="gray"), plt.xticks([]), plt.yticks([]) # 이미지 출력
plt.show()
"""
### 이미지 선명하게 하기 ###
"""
image = cv2.imread("./Long_cat.jpg",cv2.IMREAD_GRAYSCALE) # 흑백 이미지로 로드

kernel = np.array([[0,-1,0],
                   [-1,5,-1],
                   [0,-1,0]]) #커널 생성

image_sharp = cv2.filter2D(image, -1, kernel)

plt.imshow(image_sharp, cmap = "gray"), plt.axis('off') # 이미지 출력
plt.show()
"""
### 이미지 대비 높이기 ###
"""
image = cv2.imread("./Long_cat.jpg",cv2.IMREAD_GRAYSCALE) # 흑백 이미지로 로드
image_enhanced = cv2.equalizeHist(image) # 이미지 대비를 향상
plt.imshow(image_enhanced, cmap="gray"), plt.axis('off') # 이미지 출력
plt.show()

image_bgr = cv2.imread("./Long_cat.jpg") # 이미지 로드
image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV) # YUV로 변환
image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0]) # 히스토그램 평활화를 적용
image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB) # RGB로 바꿈
plt.imshow(image_rgb), plt.axis("off") # 이미지 출력
plt.show()
"""
### 색상 구분 ###
"""
image_bgr = cv2.imread("./Long_cat.jpg") # 이미지 로드
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV) # BGR에서 HSV로 변환
lower_blue = np.array([40,40,40]) # HSV에서 파랑 값의 범위를 정의
upper_blue = np.array([200,252,253])
mask = cv2.inRange(image_hsv, lower_blue, upper_blue) # 마스크 생성
image_bgr_masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask) # 이미지에 마스크를 적용
iamge_rgb = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB) # BGR에서 RGB로 변환

plt.imshow(iamge_rgb), plt.axis('off') # 이미지를 출력
plt.show()

plt.imshow(mask, cmap='gray'), plt.axis('off') # 마스크 출력
plt.show()
"""
### 이미지 이진화 ###
image_grey = cv2.imread("./Long_cat.jpg", cv2.IMREAD_GRAYSCALE) # 흑백 이미지 로드
max_output_value = 255
neighborhood_size = 99
subtract_from_mean = 10
image_binarized = cv2.adaptiveThreshold(image_grey, max_output_value,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                        neighborhood_size, subtract_from_mean) # 적응적 임계처리 적용
plt.imshow(image_binarized, cmap='gray'), plt.axis('off') # 이미지 출력
plt.show()
