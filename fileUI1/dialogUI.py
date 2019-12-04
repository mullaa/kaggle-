from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys

from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow

import file_dialog
from mask_rcnn_model import *

class fileUI(QMainWindow, file_dialog.Ui_MainWindow):
    def __init__(self, parent=None):
        super(fileUI, self).__init__(parent)
        self.setupUi(self)
        self.model = mask_rcnn_model()
        
        self.pushButton_3.clicked.connect(self.run_model)

    def run_model(self):
        self.model.run_model()
        
        
app = QApplication(sys.argv)
picture = fileUI()
picture.show()
app.exec_()
