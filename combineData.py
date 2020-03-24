
import numpy as np
import cv2
import os

pathA = os.getcwd()+'/trainA'
pathB = os.getcwd()+'/trainB'
pathAB = os.getcwd()+'/train'
filesA= os.listdir(pathA)
filesB= os.listdir(pathB)
imgPathsA = []
imgPathsB = []
for fileA in filesA:
	if not os.path.isdir(fileA) and fileA.endswith(('.bmp','.gif','.jpg','.png')):
		imgPathsA.append(pathA +"/" + fileA)

for fileB in filesB:
	if not os.path.isdir(fileB) and fileB.endswith(('.bmp','.gif','.jpg','.png')):
		imgPathsB.append(pathB+"/" + fileB)

imgPaths = zip(imgPathsA, imgPathsB)
i=0
for imgPathA, imgPathB in imgPaths:
	imgArrA = cv2.imread(imgPathA, cv2.IMREAD_GRAYSCALE)
	imgArrA = cv2.merge([imgArrA, imgArrA, imgArrA])
	imgArrA = cv2.resize(imgArrA, (256, 256), interpolation=cv2.INTER_AREA)
	imgArrB = cv2.imread(imgPathB)
	imgArrAB= np.hstack((imgArrA, imgArrB))
	cv2.imwrite(pathAB +"/" + str(i)+'.jpg', imgArrAB)
	i+=1
