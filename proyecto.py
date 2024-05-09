'''
Univesidad Del Valle de Guatemala
Visión por Computadora

Integrantes:
- Diego Córdova   20212
- Paola De León   20361
- Paola Contreras 20213
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from scipy.signal import correlate2d, convolve2d
from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse

from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize
from PIL import Image

from time import time
import cv2

from matplotlib.patches import mlines

## Leer imagen
I = cv2.imread('images/IMG_0744.jpg')
J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)


II  = np.array(I)
JJ = rgb2gray(II)

# Key points
sift = cv2.SIFT_create()
kp = sift.detect(J, None)


img = cv2.drawKeypoints(J, kp, I)
Img = np.array(img)

# Matcher

keypoints1, descriptors1 = sift.detectAndCompute(J, None)
keypoints2, descriptors2 = sift.detectAndCompute(J, None)

def getKeyPoints(img1, img2):
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    return keypoints1, descriptors1, keypoints2, descriptors2

# Crear el descriptor
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

# Mostrar los matches en la pantalla
k = 10
matched_img = cv2.drawMatches(
    J, keypoints1, J, keypoints2, matches[:k], J, flags=2
)


M = cv2.getPerspectiveTransform(keypoints1, keypoints2)
dst = cv2.warpPerspective(img, M, (300,300))

plt.imshow(dst)
plt.show()

