
import numpy as np
import cv2
import os

pathA = os.getcwd()+'/trainA'
pathB = os.getcwd()+'/trainB'
pathAB = os.getcwd()+'/train'
filesAB = os.listdir(pathAB)

for fileAB in filesAB:
	if not os.path.isdir(fileAB) and fileAB.endswith(('.bmp','.gif','.jpg','.png')):
		imgPathsAB = (pathAB +"/" + fileAB)
		imgArrAB = cv2.imread(imgPathsAB)
		imgArrA, imgArrA= np.split(imgArrAB, [256], axis=1)
		cv2.imwrite(pathA +"/" +fileAB, imgArrA)
		cv2.imwrite(pathB +"/" +fileAB, imgArrB)