import cv2
import numpy as np
import os


protofile = '/home/raj/development/facerecognition/pretrained/deploy.prototxt'
modelpath = '/home/raj/development/facerecognition/pretrained/res10_300x300_ssd_iter_140000.caffemodel'
face_embedder = '/home/raj/development/facerecognition/openface_nn4.small2.v1.t7'

IMAGE = '/home/raj/development/facerecognition/Vidyadhar'

images = os.listdir(IMAGE)
net = cv2.dnn.readNetFromCaffe(protofile, modelpath)
img_id = 1
for img in images:
    image = os.path.join(IMAGE, img)
    image = cv2.imread(image)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (xmin, ymin, xmax, ymax) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = ymin - 10 if ymin - 10 > 10 else ymin + 10
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax),(0, 255, ), 2)
    cv2.imwrite('image'+str(img_id) +'.jpg', image)
    #cv2.imshow("Output", image)
    img_id += 1
    #cv2.waitKey(0)

cv2.destroyAllWindows()
