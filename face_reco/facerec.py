import cv2
import numpy as np
import pickle
import os
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

'''
data needed for face recognition
1. dataset - images
2. prototxt file - config file
3. trained model - weights

'''

class Facerecognition:

    def __init__(self, dataset, detector, output, pretrained_embedding_model):

        self.dataset = dataset
        self.prototxt = detector
        self.face_detector = detector
        self.output_embeddings = os.path.join(output, 'embeddings.pickle')
        self.pretrained_embedding_model = pretrained_embedding_model
        self.trained_recognizer = os.path.join(output, 'recognizer.pickle')
        self.label_encoder = os.path.join(output, 'le.pickle')

    def loadModels(self):
        '''
            loads serialized face detector and embedding model.
        '''
        prototxt_path = os.path.join(self.prototxt, 'deploy.prototxt')
        model_path = os.path.join(self.face_detector, 'res10_300x300_ssd_iter_140000.caffemodel')
        detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        embedder = cv2.dnn.readNetFromTorch(self.pretrained_embedding_model)
        return detector, embedder

    def extractFaceEmbeddings(self):
        '''
            loops over the image dataset and extracts face embeddings.
        '''
        base_dir = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(base_dir, self.dataset)
        labels = []
        image_paths = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg") or file.endswith("JPG"):
                    path = os.path.join(root, file)
                    image_paths.append(path)
                    label = os.path.basename(os.path.dirname(path))
                    labels.append(label)

        labels = set(labels)
        facial_embeddings = []
        names = []

        detector, embedder = self.loadModels()

        for i, imagepath in enumerate(image_paths):
            name = imagepath.split(os.path.sep)[-2]
            image = cv2.imread(imagepath)
            r = 600.0 / image.shape[1]
            dim = (600, int(image.shape[0] * r))
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            (h, w) = image.shape[:2]
            imageblob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                            (104.0, 177.0, 123.0), swapRB=False, crop=False)
            
            detector.setInput(imageblob)
            detections = detector.forward()
            if len(detections) > 0:
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (minx, miny, maxx, maxy) = box.astype("int")
                    face = image[miny:maxy, minx:maxx]
                    (face_h, face_w) = face.shape[:2]
                    if face_w < 20 or face_h < 20:
                        continue
                    faceblob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                            (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceblob)
                    vec = embedder.forward()
                    names.append(name)
                    facial_embeddings.append(vec.flatten())
        print("[info] serializing encodings...")
        data = {"embeddings": facial_embeddings, "names": names}
        f = open(self.output_embeddings, "wb")
        f.write(pickle.dumps(data))

    def faceTrainer(self):
        data = pickle.loads(open(self.output_embeddings, "rb").read())
        print('data in trainer', data)
        le = LabelEncoder()
        labels = le.fit_transform(data["names"])

        print("[info] training model...")
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(data["embeddings"], labels)

        # write the actual face recognition model to disk
        f = open(self.trained_recognizer, "wb")
        f.write(pickle.dumps(recognizer))
        f.close()

        # write the label encoder to disk
        f = open(self.label_encoder, "wb")
        f.write(pickle.dumps(le))
        f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required = True)
    parser.add_argument('--detector', required = True)
    parser.add_argument('--output_embeddings', required = True)
    parser.add_argument('--model', required = True)
    args = vars(parser.parse_args())

    fr = Facerecognition(args['dataset'], args['detector'], args['output_embeddings'], args['model'])
    fr.extractFaceEmbeddings()
    print('training model...')
    fr.faceTrainer()
