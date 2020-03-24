#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os ,math
import cv2
import numpy as np
from PIL import Image

import PySide2
from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtOpenGL import *
from PySide2.QtWidgets import QApplication, QLabel, QFileDialog, QMainWindow
from main_dialog_ui import Ui_MainDialog

from OpenGL.GL import *
from OpenGL.GL import shaders

lib_path = os.path.abspath(os.path.join('./dp'))
sys.path.append(lib_path)
from _test import _test

model_name = 'depth_transfer'
model_type = 'cycle_gan'
model_netG = 'unet_256'
model_path = './dp/checkpoints'
dataset_path = './dp/datasets/temp'
result_path = './results/'
norm_set = 'batch'

model_depth = _test(model_name, model_type, model_netG, model_path, dataset_path, result_path, norm_set)
model_jet2gray = _test('jet2gray_pix2pix', 'pix2pix', 'unet_256', './dp/checkpoints', './dp/datasets/temp_jet2gray', result_path, 'batch')

def dpEntryCyclegan(image):
	imgArr = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
	imgArr = cv2.merge([imgArr, imgArr, imgArr])
	imgArr = cv2.resize(imgArr, (256, 256), interpolation=cv2.INTER_AREA)
	blank = np.ones((256,256,3))
	imgArr = np.hstack((imgArr, blank))
	cv2.imwrite(dataset_path+'/test/image.jpg',imgArr)
	model_depth.doTest()

def dpEntryJet2Gray():
	depthJetArr = cv2.imread(result_path + model_name+'/test_latest/images/image_fake_B.png', cv2.IMREAD_UNCHANGED)
	blank = np.ones((256,256,3))
	depthJetArr = np.hstack((depthJetArr, blank))
	cv2.imwrite('./dp/datasets/temp_jet2gray/test/image.jpg',depthJetArr)
	model_jet2gray.doTest()

def shaderFromFile(shaderType, shaderFile):
	shaderSrc = ''
	with open(shaderFile) as sf:
		shaderSrc = sf.read()
	return shaders.compileShader(shaderSrc, shaderType)

def blurImage(image, depth, radius, sigma, focus, dof, depthContrast, drad, dsig):
	imgArr = cv2.imread(image, cv2.IMREAD_UNCHANGED)
	imgArr = cv2.cvtColor(imgArr,cv2.COLOR_BGR2RGB)
	if imgArr.shape[0]>1000:
		scale = 1000/imgArr.shape[0]
		imgArr = cv2.resize(imgArr, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
	elif imgArr.shape[1]>1000:
		scale = 1000/imgArr.shape[1]
		imgArr = cv2.resize(imgArr, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
	blur = cv2.GaussianBlur(imgArr,(radius*2+1,radius*2+1),sigma,cv2.BORDER_DEFAULT)

	depthGrayArr = cv2.imread(depth, cv2.IMREAD_UNCHANGED)
	depthGrayArr = depthGrayArr*(depthContrast/127+1)-depthContrast
	depthGrayArr[depthGrayArr>255]=255
	depthGrayArr[depthGrayArr<0]=0
	depthGrayArr = np.uint8(depthGrayArr)
	depthGrayArr = cv2.GaussianBlur(depthGrayArr,(drad*2+1,drad*2+1),dsig,cv2.BORDER_DEFAULT)

	mix = depthGrayArr/255
	newMix = np.zeros(mix.shape)
	nears=-dof*2/2+focus
	fars=dof*2/2+focus
	for i in range(mix.shape[0]):
		for j in range(mix.shape[1]):
			if mix[i,j,0]>=nears and mix[i,j,0]<=fars:
				newMix[i,j]=(0, 0, 0)
			elif mix[i,j,0]<nears:
				newMix[i,j]=np.abs(mix[i,j]-nears)
			elif mix[i,j,0]>fars:
				newMix[i,j]=np.abs(mix[i,j]-fars)

	if not newMix.max()==newMix.min():
			newMix=(newMix-newMix.min())/(newMix.max()-newMix.min())
	
	depthGrayArr = cv2.resize(depthGrayArr, (imgArr.shape[1], imgArr.shape[0]), interpolation=cv2.INTER_CUBIC)
	newMix = cv2.resize(np.uint8(newMix*255), (imgArr.shape[1], imgArr.shape[0]), interpolation=cv2.INTER_CUBIC)/255
	res = np.uint8(imgArr*(1-newMix)+blur*newMix)
	ns= np.hstack((imgArr, depthGrayArr ,res))
	return Image.fromarray(ns)

class MainWindow(QMainWindow):
	def __init__(self, gformat):
		super(MainWindow, self).__init__()
		self.imgWidget = None
		self.ui = Ui_MainDialog()
		self.ui.setupUi(self)
		self.ui.FileButton.clicked.connect(self.loadFile)
		self.ui.KSizeSpin.valueChanged.connect(self.changePars)
		self.ui.SigmaSpin.valueChanged.connect(self.changePars)
		self.ui.FocalLenSpin.valueChanged.connect(self.changePars)
		self.ui.DOFSpin.valueChanged.connect(self.changePars)
		self.ui.ContSpin.valueChanged.connect(self.changePars)
		self.ui.DKSizeSpin.valueChanged.connect(self.changePars)
		self.ui.DSigmaSpin.valueChanged.connect(self.changePars)

	def closeEvent(self,event):
		if not self.imgWidget==None:
			self.imgWidget.close()
			##clean temp
	@Slot()
	def changePars(self):
		if not self.imgWidget==None:
			self.imgWidget.radius = self.ui.KSizeSpin.value()
			self.imgWidget.sigma = self.ui.SigmaSpin.value()
			self.imgWidget.focus = self.ui.FocalLenSpin.value()
			self.imgWidget.dof = self.ui.DOFSpin.value()
			self.imgWidget.depthContrast = self.ui.ContSpin.value()
			self.imgWidget.drad = self.ui.DKSizeSpin.value()
			self.imgWidget.dsig = self.ui.DSigmaSpin.value()
			self.imgWidget.reDraw()

	@Slot()
	def loadFile(self):
		dialog = QFileDialog()
		dialog.setFileMode(QFileDialog.AnyFile)
		dialog.setViewMode(QFileDialog.Detail)
		dialog.setNameFilter("Images (*.png *.bmp *.jpg)");
		if dialog.exec_():
			fileName = dialog.selectedFiles()[0]
			self.ui.FileLineEdit.setText(fileName)
			self.imgWidget = MyGLWidget(gformat, fileName,	self.ui.KSizeSpin.value(), 
															self.ui.SigmaSpin.value(), 
															self.ui.FocalLenSpin.value(),
															self.ui.DOFSpin.value(),
															self.ui.ContSpin.value(),
															self.ui.DKSizeSpin.value(), 
															self.ui.DSigmaSpin.value())
			self.imgWidget.show()

class MyGLWidget(QGLWidget):
	def __init__(self, gformat, imgFile, radius, sigma, focus, dof, depthContrast, drad, dsig, parent=None):
		super(MyGLWidget, self).__init__(gformat, parent)
		
		self.vaoID = None
		self.vboVerticesID = None
		self.vboIndicesID = None
		self.textureID = None
		self.sprogram = None
		
		self.vertices = None
		self.indices = None

		self.imgFile = imgFile

		self.radius=radius
		self.sigma=sigma
		self.focus=focus
		self.dof = dof

		self.depthContrast = depthContrast
		self.drad = drad
		self.dsig = dsig

		dpEntryCyclegan(self.imgFile)
		dpEntryJet2Gray()

		self.__setImage()
		self.setGeometry(40, 40, 1000, self.im.size[1]*(1000/self.im.size[0]))
		self.setWindowTitle('Blur')

	def __setImage(self):
		self.im = blurImage(self.imgFile,result_path + 'jet2gray_pix2pix/test_latest/images/image_fake_B.png', 
							self.radius, self.sigma, self.focus, self.dof, 
							self.depthContrast,  self.drad, self.dsig)
	def __setShader(self):
		vshader = shaderFromFile(GL_VERTEX_SHADER, './shader/shader.vert')
		fshader = shaderFromFile(GL_FRAGMENT_SHADER, './shader/shader.frag')
		self.sprogram = shaders.compileProgram(vshader, fshader)
		glUseProgram(self.sprogram)
		self.vertexAL = glGetAttribLocation(self.sprogram, 'pos')
		self.tmUL = glGetUniformLocation(self.sprogram, 'textureMap')
		glUniform1i(self.tmUL, 0)
		glUseProgram(0)

	def __setVBO(self):
		self.vertices = np.array((0.0, 0.0, 
								  1.0, 0.0, 
								  1.0, 1.0, 
								  0.0, 1.0), dtype=np.float32)
		self.indices = np.array((0, 1, 2, 
								 0, 2, 3), dtype=np.ushort)
		self.vaoID = glGenVertexArrays(1)
		self.vboVerticesID = glGenBuffers(1)
		self.vboIndicesID = glGenBuffers(1)
		glBindVertexArray(self.vaoID)
		glBindBuffer(GL_ARRAY_BUFFER, self.vboVerticesID)
		glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
		glEnableVertexAttribArray(self.vertexAL)
		glVertexAttribPointer(self.vertexAL, 2, GL_FLOAT, GL_FALSE, 0, None)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboIndicesID)
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

	def __setTexture(self):
		# flip the image in the Y axis
		im = self.im.transpose(Image.FLIP_TOP_BOTTOM)
		self.textureID = glGenTextures(1)
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, self.textureID)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.im.size[0], self.im.size[1], 
					 0, GL_RGB, GL_UNSIGNED_BYTE, im.tobytes())

	def initializeGL(self):
		glClearColor(0, 0, 0, 0)
		self.__setShader()
		self.__setVBO()
		self.__setTexture()

	def resizeGL(self, w, h):
		scale=1
		scaleH = h/self.im.size[1]
		scaleW = w/self.im.size[0]
		if scaleH*self.im.size[0]>w: scale = scaleW
		else: scale = scaleH
		if scaleW*self.im.size[1]>h: scale = scaleH
		else: scale = scaleW
		glViewport(0, h-int(scale*self.im.size[1]), int(scale*self.im.size[0]), int(scale*self.im.size[1]))

	def paintGL(self, *args, **kwargs):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		glUseProgram(self.sprogram)
		glDrawElements(GL_TRIANGLES, self.indices.size, GL_UNSIGNED_SHORT, None)
		glUseProgram(0)

	def reDraw(self):
		self.__setImage()
		self.__setTexture()
		self.paintGL()
		self.swapBuffers()

if __name__ == '__main__':
	app = QApplication(sys.argv)
	gformat = QGLFormat()
	mainWindow = MainWindow(gformat)
	mainWindow.show()
	
	sys.exit(app.exec_())

