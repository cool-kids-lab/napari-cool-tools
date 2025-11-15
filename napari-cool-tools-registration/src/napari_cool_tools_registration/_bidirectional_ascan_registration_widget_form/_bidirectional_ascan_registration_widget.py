# This Python file uses the following encoding: utf-8

from qtpy.QtWidgets import QDialog
from qtpy import QtWidgets
from _bidirectional_ascan_registration_form import Ui_Dialog
import pyqtgraph as pg
import numpy as np

class Bidirectional_Ascan_Registration_Widget(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Bidirectional Ascan Registration Dialog")
        # Create pyqtgraph image viewer
        self.viewer = pg.ImageView(parent=self)
        self.viewer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.viewer.updateGeometry()

        # Create pyqtgraph plot viewer
        self.plotter = pg.PlotWidget()
        self.plotter.setSizePolicy(self.sizePolicy())
        self.plotter.setBackground('w')

        layout = self.graphicsViewPlaceHolder.parent().layout()
        layout.replaceWidget(self.graphicsViewPlaceHolder, self.viewer)
        self.graphicsViewPlaceHolder.deleteLater()

        self.viewer.ui.roiBtn.hide()
        self.viewer.ui.menuBtn.hide()
        self.viewer.ui.histogram.hide()

        image = np.random.rand(256, 256) * 255
        self.viewer.setImage(image.astype(np.float32))

    def updateImage(self):
        x = np.random.rand(256, 256) * 255
        self.viewer.setImage(x.astype(np.float32))
