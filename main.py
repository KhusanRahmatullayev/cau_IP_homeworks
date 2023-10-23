import cv2
import numpy as np
import matplotlib.pyplot as plt
# task 1 for gray_scale images
# read color image
image = cv2.imread('GrayScale.jpg')
cv2.imshow('Original', image)
cv2.waitKey(3000)

# change image from color to the gray_scale format
gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# claculate histogram of gray_scale image and output it
hist = cv2.calcHist([gray_scale], [0], None, [256], [0, 256])
plt.plot(hist)
plt.title("Gray_Scale image hist")
plt.show()

# equlaized function for gray_scale images
eq_gray_scale = cv2.equalizeHist(gray_scale)

# after equalized histogram of gray_scale image and plot it
# with title Equalized
eq_hist = cv2.calcHist([eq_gray_scale], [0], None, [256], [0, 256])
plt.plot(eq_hist)
plt.title('Equalized')
plt.show()

# task 2 for color images
# read color image
image_color = cv2.imread('RGB.jpg')
cv2.imshow("original", image_color)
cv2.waitKey(3000)

# calaculate histogram of color image and output it
hist_color = cv2.calcHist([image_color], [0], None, [256], [0, 256])
plt.plot(hist_color)
plt.title("Color image hist before eq")
plt.show()

# equlaized function for color images
channels = cv2.split(image_color)
eq_channels = []
for ch, color in zip(channels, ['B', 'G', 'R']):
    eq_channels.append(cv2.equalizeHist(ch))
eq_image_color = cv2.merge(eq_channels)
eq_image_color = cv2.cvtColor(eq_image_color, cv2.COLOR_BGR2RGB)
cv2.imshow("Equalized_color Image hist", eq_image_color)
cv2.waitKey(5000)

# after equalized histogram of gray_scale image and plot it
# with title Equalized_color Image
# show Histogram
channels = ('b', 'g', 'r')
# we now separate the colors and plot each in the Histogram
for i, color in enumerate(channels):
    histogram = cv2.calcHist([eq_image_color], [i], None, [256], [0, 256])
    plt.plot(histogram)
    plt.xlim([0, 256])
plt.show()