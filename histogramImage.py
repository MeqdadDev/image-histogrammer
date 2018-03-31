### Importing statements
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Reading the images

# First image
img = cv2.imread('palestine.jpg',0)
cv2.imshow('PalestineGray',img)
cv2.waitKey(0)
cv2.destroyWindow('PalestineGray')

# Second image
img2 = cv2.imread('syria.jpg',0)
cv2.imshow('SyriaGray',img2)
cv2.waitKey(0)
cv2.destroyWindow('SyriaGray')

### There are different ways to calculate the histogram for an image.

# First method... using OpenCV functions [ calcHist() ]
histOpenCV = cv2.calcHist([img],[0],None,[256],[0,256])

# Second method... using [histogram] in (nmupy) Numerical Python library.
histOpenCV, histNumPy = np.histogram(img.ravel(),256,[0,256])

# Third method... using "bincount" in (nmupy) Numerical Python library.
histBinCount = np.bincount(img.ravel(),minlength=256)

## Note: OpenCV function is more faster than (around 40X) than np.histogram(). "from opencv Doc."

### Showing the results....

# Printing it as array...
print(".....::::: Histogram using OpenCV function:::::.....\n",histOpenCV)
print(".....::::: Histogram using NumPy function [np.histogram]:::::.....\n",histNumPy)
print(".....::::: Histogram using NumPy function [np.bincount]:::::.....\n",histBinCount)

# Plotting it, using matplotlib with different ways...
# First histogram
plt.hist(img.ravel(),256,[0,256])
plt.title('Palestine Gray Histogram')
plt.show()

# Second histogram
plt.hist(img2.ravel(),256,[0,256])
plt.title('Syria Gray Histogram')
plt.show()

# Normal Plot of matplotlib
color = ('b', 'g', 'r')

# First image
for i, col in enumerate(color):
    histOpenCV = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(histOpenCV, color=col)
    plt.xlim([0, 256])
plt.title("[Palestine] Normal Plot of matplotlib")
plt.show()

# Second image
for i, col in enumerate(color):
    histOpenCV2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    plt.plot(histOpenCV2, color=col)
    plt.xlim([0, 256])
plt.title("[Syria] Normal Plot of matplotlib")
plt.show()

### More features will be added soooooon.
#### Thanks ####

