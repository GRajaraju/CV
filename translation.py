
# Image translation is basically shifting image by certain distance. Consider a point on xy plane, if we want to shift this
# points by 40 units horizontally and 40 units vertically, we specify such translation as 
#           x2 = x1 + dx where dx = 40
#           y2 = y1 + dy where dy = 40
# The above expressions can be represented in a matrix form as
#            [x2,y2,1] = [[1,0,dx],[0,1,dy]] [x1,y1,1]
#     

import cv2
import numpy as np

image = cv2.imread('bird.jpg')
rows,cols = image.shape[:2]
translation_matrix = np.float32([1,0,40],[0,1,40]) # shifting by 40 units in right and 40 in down. 
image_translated = cv2.warpAffine(image,translation_matrix,(cols+40,rows+40),cv2.INTER_LINEAR) #adding 40 units to avoid image crop
cv2.imshow('Image Translation',image_translated)
cv.waitKey()
cv2.destroyAllWindows()

