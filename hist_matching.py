# import packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

# # task 6
# # read image 1
# img1 = cv2.imread("RGB.jpg")
# cv2.imshow('Original', img1)
#
# # checking the number of channels
# print('No of Channel is: ' + str(img1.ndim))
#
# # I take second image which is grayScaled of first image
# img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Image change one', img2)
#
# # checking the number of channels
# print('No of Channel is: ' + str(img1.ndim))
#
# matched = match_histograms(img1, img1)
#
# # checkout out match image
# cv2.imshow("Matched", matched)
# cv2.waitKey(10000)

# task 7 matching I_ref image and I_log image
# i_ref = cv2.imread('GrayScale.jpg')
i_ref = cv2.imread('RGB.jpg')

# Apply log transformation method
c = 255 / np.log(1 + np.max(i_ref))
i_log = c * (np.log(i_ref + 1))
i_log = np.array(i_log, dtype=np.uint8)

i_matched = match_histograms(i_ref, i_log)

cv2.imshow("I_Ref image", i_ref)
cv2.imshow("I_Log image", i_log)
cv2.imshow("I_Matched image", i_matched)

cv2.waitKey(8000)
# If I applied log transformation and match them, I will get differant output
# cause it depend on the pictures.

# task 8
def myHistMatch(image1, image2):
    matched_image = match_histograms(image1, image2)
    return matched_image

