# A simple program to find dimensions of an object in real time. The dimensions of the object calculated in this program are 
# with respect to the image size.
#
import sys
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:

  ret,frame_original = cap.read()
  frame = cv2.cvtColor(frame_orginial,cv2.COLOR_BGR2GRAY)
  frame = cv2.GaussianBlur(frame,(5,5),0)
  ret1,frame = cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  frame = cv2.Laplacian(frame,cv2.CV_8UC1)
  _,contours, hierarchy = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  contour = contours[0]
  x,y,w,h = cv2.boundingRect(contour)
  frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
  rect = cv2.minAreaRect(contour)
  box = cv2.boxPoints(rect)
  box = np.int0(box)
  cv2.rectangle(frame_orginial,(x,y),(x+w,y+h),(0,255,0),2)
  cv2.putText(frame_orginial,"Dimensions="+str(h) + "x" + str(w),(int(w/2),int(h/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0))
  cv2.imshow("Object Dimensions",frame_orginial)
  cv2.waitKey(0)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    cap.release()
    cv2.destroyAllWindows()
    break




