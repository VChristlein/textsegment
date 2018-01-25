import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('/tmp/dibco/dibco/gt/2011-001-h.png', 0)
print(img.shape)
plt.imshow(img[80:150, :100], cmap='gray')
plt.show()
img = cv2.bitwise_not(img)
res = cv2.distanceTransform(img, cv2.DIST_L2, 5)
res = np.where(res >= 2, 1 - 1 / res, 0)
print(np.max(res))
plt.imshow(res[80:150, :100], cmap='gray')
plt.show()

img = cv2.imread('/tmp/dibco/dibco/gt/2011-001-h.png', 0)
res = cv2.distanceTransform(img, cv2.DIST_L2, 3)
plt.imshow(res[80:150, :100], cmap='gray')
plt.show()
res = np.where(res >= 8, 1, np.where(res == 0, 1, 1 + res / 8))
print(np.max(res))
print(np.min(res))
plt.imshow(res[80:150, :100], cmap='gray')
plt.show()
exit(1)
