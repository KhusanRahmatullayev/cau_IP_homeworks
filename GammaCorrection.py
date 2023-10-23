import cv2
import numpy as np
import matplotlib.pyplot as plt
# task 3
def histGammaCorr(image, gamma):
    table = [((i/255)** (1/gamma)) * 255 for i in range(0, 256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(image, table)

img = cv2.imread('GrayScale.jpg')
gammaImg = histGammaCorr(img, 2.2)

cv2.imshow('Original image', img)
cv2.imshow('Gramma Correction image',gammaImg)
cv2.waitKey(7000)


# task 4
# histogram for gamma = 0.2
img_g_1 = histGammaCorr(img, 0.2)
gray_scale_1 = cv2.cvtColor(img_g_1, cv2.COLOR_BGR2GRAY)
hist_1 = cv2.calcHist([gray_scale_1], [0], None, [256], [0, 256])
plt.plot(hist_1)
plt.title("Gray_Scale image hist 1")
plt.show()

# histogram for gamma = 2
img_g_2 = histGammaCorr(img, 2)
gray_scale_2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist_2 = cv2.calcHist([gray_scale_2], [0], None, [256], [0, 256])
plt.plot(hist_2)
plt.title("Gray_Scale image hist 2")
plt.show()

# task 5 when we apply many times histEqual
# function the outputs same as performing it once
for i in range (0, 3):
    gray_scale_2 = cv2.equalizeHist(gray_scale_2)
hist_2 = cv2.calcHist([gray_scale_2], [0], None, [256], [0, 256])
plt.plot(hist_2)
plt.title("Gray_Scale image hist 2 after 3 times")
plt.show()

