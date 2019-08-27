import cv2
import os


def data_folder():

	path = '/home/user/Development/face_recognition/dataset'
	folder = input('enter name: ')
	image_path = os.path.join(path, folder)
	if not os.path.exists(image_path):
		new_path = os.mkdir(image_path)
	return image_path



image_path = data_folder()
cap = cv2.VideoCapture('20181228_153823.mp4')
total_frames = 0
if cap:
	print('live streaming')

while (total_frames < 400):
	_, frame = cap.read()

	cv2.imwrite(os.path.join(image_path, 'image' + str(total_frames) + '.jpg'), frame)
	total_frames += 1
	cv2.imshow('image', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
