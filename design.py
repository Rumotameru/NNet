from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
import os
import torch
import visualization
import network
import data_loader as dl
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as Navigation



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1271, 744)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.SamplePreview = QtWidgets.QLabel(self.centralwidget)
        self.SamplePreview.setGeometry(QtCore.QRect(30, 70, 531, 541))
        self.SamplePreview.setText("")
        self.SamplePreview.setPixmap(QtGui.QPixmap("sample.png"))
        self.SamplePreview.setObjectName("SamplePreview")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(580, 220, 551, 71))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setTextFormat(QtCore.Qt.AutoText)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.path_enter = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.path_enter.setObjectName("path_enter")
        self.horizontalLayout_2.addWidget(self.path_enter)
        self.file_browser = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.file_browser.setObjectName("file_browser")
        self.horizontalLayout_2.addWidget(self.file_browser)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(580, 90, 676, 80))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.Title = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.Title.setFont(font)
        self.Title.setObjectName("Title")
        self.verticalLayout.addWidget(self.Title)
        self.subtitle = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setItalic(True)
        self.subtitle.setFont(font)
        self.subtitle.setObjectName("subtitle")
        self.verticalLayout.addWidget(self.subtitle)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(30, 620, 341, 81))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_5 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_2.addWidget(self.label_5)
        self.label_6 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_2.addWidget(self.label_6)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(660, 340, 391, 301))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.checkButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkButton.sizePolicy().hasHeightForWidth())
        self.checkButton.setSizePolicy(sizePolicy)
        self.checkButton.setObjectName("checkButton")
        self.horizontalLayout_3.addWidget(self.checkButton)
        self.widget = QtWidgets.QWidget(self.horizontalLayoutWidget_2)
        self.widget.setObjectName("widget")
        self.horizontalLayout_3.addWidget(self.widget)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.file_browser.clicked.connect(self.browse_files)
        self.checkButton.clicked.connect(self.print_result)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.layout = QtWidgets.QVBoxLayout(self.widget)
        self.layout.addWidget(self.canvas)
        self.counter = 0


    def browse_files(self):
        imagePath = QFileDialog.getOpenFileName(None, 'Open', os.getcwd(), "Bitmap Picture files (*.bmp)")
        imagePath = imagePath[0]
        self.path_enter.setText(imagePath)

    def predict_label(self):
        model = network.CNN()
        model.load_state_dict(torch.load("model_state.pt"))
        model.cuda()
        model.eval()
        image = dl.prediction_data(self.path_enter.text())
        for pic in image:
            pic = pic.cuda()
            output = model(pic)
            _, predicted = torch.max(output.data, 1)
            return predicted.cpu().numpy()[0]

    def print_result(self):
        class_names = ["Normal", "ALL"]
        colors = ["b", "r"]
        file = self.path_enter.text()
        ax = self.canvas.figure.add_subplot(111)
        label = self.predict_label()
        img = plt.imread(file)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(class_names[label], fontsize=28, color=colors[label])
        self.canvas.draw()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Leukimia Detector"))
        self.label_2.setText(_translate("MainWindow", "Cell image"))
        self.file_browser.setText(_translate("MainWindow", "Browse..."))
        self.Title.setText(_translate("MainWindow", "<html><head/><body><p align=\"justify\"><span style=\" color:#ff0000;\">Find out if there any acute lymphoblastic leukemia by blood cell\'s photo</span></p></body></html>"))
        self.subtitle.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" color:#c78500;\">Accuracy is about 72,1% !!!</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">ALL - cells with acute lymphoblastic leukemia</span></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">Normal - cells without leukimia</span></p></body></html>"))
        self.checkButton.setText(_translate("MainWindow", "Check"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
