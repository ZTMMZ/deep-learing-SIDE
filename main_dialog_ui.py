# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\main_dialog_ui.ui',
# licensing of '.\main_dialog_ui.ui' applies.
#
# Created: Wed Mar 20 16:52:54 2019
#      by: pyside2-uic  running on PySide2 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_MainDialog(object):
	def setupUi(self, MainDialog):
		MainDialog.setObjectName("MainDialog")
		MainDialog.resize(350, 381)
		self.BlurParGroupBox = QtWidgets.QGroupBox(MainDialog)
		self.BlurParGroupBox.setGeometry(QtCore.QRect(10, 50, 331, 171))
		self.BlurParGroupBox.setObjectName("BlurParGroupBox")
		self.KSizeSpin = QtWidgets.QSpinBox(self.BlurParGroupBox)
		self.KSizeSpin.setGeometry(QtCore.QRect(180, 26, 70, 20))
		self.KSizeSpin.setMinimum(1)
		self.KSizeSpin.setMaximum(256)
		self.KSizeSpin.setObjectName("KSizeSpin")
		self.SigmaSpin = QtWidgets.QDoubleSpinBox(self.BlurParGroupBox)
		self.SigmaSpin.setGeometry(QtCore.QRect(180, 64, 70, 20))
		self.SigmaSpin.setMaximum(256.0)
		self.SigmaSpin.setSingleStep(0.1)
		self.SigmaSpin.setProperty("value", 1.0)
		self.SigmaSpin.setObjectName("SigmaSpin")
		self.FocalLenSpin = QtWidgets.QDoubleSpinBox(self.BlurParGroupBox)
		self.FocalLenSpin.setGeometry(QtCore.QRect(180, 98, 70, 20))
		self.FocalLenSpin.setMinimum(-1.0)
		self.FocalLenSpin.setMaximum(2.0)
		self.FocalLenSpin.setSingleStep(0.01)
		self.FocalLenSpin.setObjectName("FocalLenSpin")
		self.KSizeLabel = QtWidgets.QLabel(self.BlurParGroupBox)
		self.KSizeLabel.setGeometry(QtCore.QRect(70, 30, 81, 16))
		self.KSizeLabel.setObjectName("KSizeLabel")
		self.SigmaLabel = QtWidgets.QLabel(self.BlurParGroupBox)
		self.SigmaLabel.setGeometry(QtCore.QRect(70, 65, 71, 16))
		self.SigmaLabel.setObjectName("SigmaLabel")
		self.FocalLenLabel = QtWidgets.QLabel(self.BlurParGroupBox)
		self.FocalLenLabel.setGeometry(QtCore.QRect(70, 100, 81, 16))
		self.FocalLenLabel.setObjectName("FocalLenLabel")
		self.DOFLabel = QtWidgets.QLabel(self.BlurParGroupBox)
		self.DOFLabel.setGeometry(QtCore.QRect(70, 135, 91, 16))
		self.DOFLabel.setObjectName("DOFLabel")
		self.DOFSpin = QtWidgets.QDoubleSpinBox(self.BlurParGroupBox)
		self.DOFSpin.setGeometry(QtCore.QRect(180, 134, 70, 20))
		self.DOFSpin.setDecimals(2)
		self.DOFSpin.setMinimum(0.0)
		self.DOFSpin.setMaximum(1.0)
		self.DOFSpin.setSingleStep(0.01)
		self.DOFSpin.setObjectName("DOFSpin")
		self.FileButton = QtWidgets.QToolButton(MainDialog)
		self.FileButton.setGeometry(QtCore.QRect(308, 10, 34, 22))
		self.FileButton.setObjectName("FileButton")
		self.FileLineEdit = QtWidgets.QLineEdit(MainDialog)
		self.FileLineEdit.setGeometry(QtCore.QRect(10, 11, 300, 20))
		self.FileLineEdit.setObjectName("FileLineEdit")
		self.DParGroupBox = QtWidgets.QGroupBox(MainDialog)
		self.DParGroupBox.setGeometry(QtCore.QRect(10, 235, 331, 135))
		self.DParGroupBox.setObjectName("DParGroupBox")
		self.ContSpin = QtWidgets.QDoubleSpinBox(self.DParGroupBox)
		self.ContSpin.setGeometry(QtCore.QRect(180, 26, 70, 20))
		self.ContSpin.setDecimals(1)
		self.ContSpin.setMinimum(0.0)
		self.ContSpin.setMaximum(255.0)
		self.ContSpin.setSingleStep(1.0)
		self.ContSpin.setObjectName("ContSpin")
		self.DKSizeSpin = QtWidgets.QSpinBox(self.DParGroupBox)
		self.DKSizeSpin.setGeometry(QtCore.QRect(180, 64, 70, 20))
		self.DKSizeSpin.setMinimum(1)
		self.DKSizeSpin.setMaximum(256)
		self.DKSizeSpin.setObjectName("DKSizeSpin")
		self.ContLabel = QtWidgets.QLabel(self.DParGroupBox)
		self.ContLabel.setGeometry(QtCore.QRect(70, 30, 91, 16))
		self.ContLabel.setObjectName("ContLabel")
		self.DSigmaSpin = QtWidgets.QDoubleSpinBox(self.DParGroupBox)
		self.DSigmaSpin.setGeometry(QtCore.QRect(180, 98, 70, 20))
		self.DSigmaSpin.setMaximum(256.0)
		self.DSigmaSpin.setSingleStep(0.1)
		self.DSigmaSpin.setProperty("value", 1.0)
		self.DSigmaSpin.setObjectName("DSigmaSpin")
		self.DKSizeLabel = QtWidgets.QLabel(self.DParGroupBox)
		self.DKSizeLabel.setGeometry(QtCore.QRect(70, 65, 101, 16))
		self.DKSizeLabel.setObjectName("DKSizeLabel")
		self.DSigmaLabel = QtWidgets.QLabel(self.DParGroupBox)
		self.DSigmaLabel.setGeometry(QtCore.QRect(70, 100, 91, 16))
		self.DSigmaLabel.setObjectName("DSigmaLabel")

		self.retranslateUi(MainDialog)
		QtCore.QMetaObject.connectSlotsByName(MainDialog)

	def retranslateUi(self, MainDialog):
		MainDialog.setWindowTitle(QtWidgets.QApplication.translate("MainDialog", "Main Panel", None, -1))
		self.BlurParGroupBox.setTitle(QtWidgets.QApplication.translate("MainDialog", "Shallow DOF Parameters", None, -1))
		self.KSizeLabel.setText(QtWidgets.QApplication.translate("MainDialog", "Kernel Radius", None, -1))
		self.SigmaLabel.setText(QtWidgets.QApplication.translate("MainDialog", "Sigma", None, -1))
		self.FocalLenLabel.setText(QtWidgets.QApplication.translate("MainDialog", "Focal Length", None, -1))
		self.DOFLabel.setText(QtWidgets.QApplication.translate("MainDialog", "Depth of Field", None, -1))
		self.FileButton.setText(QtWidgets.QApplication.translate("MainDialog", "...", None, -1))
		self.DParGroupBox.setTitle(QtWidgets.QApplication.translate("MainDialog", "Depth Map Augmentation", None, -1))
		self.ContLabel.setText(QtWidgets.QApplication.translate("MainDialog", "Depth Contrast", None, -1))
		self.DKSizeLabel.setText(QtWidgets.QApplication.translate("MainDialog", "Depth Kernel Rad", None, -1))
		self.DSigmaLabel.setText(QtWidgets.QApplication.translate("MainDialog", "Depth Sigma", None, -1))
