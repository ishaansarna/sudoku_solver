import cv2
from matplotlib import pyplot as plt
import numpy as np

img_path: str = '../data/sudoku-001.jpg'

img = cv2.imread(img_path, 0)

blurImg = cv2.GaussianBlur(img, (19, 19), 0)

thImg = cv2.adaptiveThreshold(blurImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 11, 2)

invImg = cv2.bitwise_not(thImg)

kernel = np.ones((5, 5), np.uint8)
dilImg = cv2.dilate(invImg, kernel, iterations=1)

plt.subplot(231), plt.imshow(img, 'gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(blurImg, 'gray'), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(thImg, 'gray'), plt.title('Thresholded')
plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(invImg, 'gray'), plt.title('Inverted')
plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(dilImg, 'gray'), plt.title('Dilated')
plt.xticks([]), plt.yticks([])
plt.show()
