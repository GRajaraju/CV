
import cv2


img = cv2.imread('img path')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def convert_binary_image(gray):
""" Convert the gray scale image to binary """

  for i in range(gray.shape[0]):
      for j in range(gray.shape[1]):
          if gray[i, j] > 127:
              gray[i, j] = 255
          else:
              gray[i, j] = 0
  return gray

