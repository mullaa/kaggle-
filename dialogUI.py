from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys

from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow

import file_dialog

class fileUI(QMainWindow, file_dialog.Ui_MainWindow):
    def __init__(self, parent=None):
        super(fileUI, self).__init__(parent)
        self.setupUi(self)

app = QApplication(sys.argv)
picture = fileUI()
picture.show()
app.exec_()