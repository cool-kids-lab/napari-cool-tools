from qtpy.QtWidgets import QDialog
from qtpy import QtWidgets
from napari_cool_tools_registration._bidirectional_ascan_registration_form import Ui_Dialog
import pyqtgraph as pg
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from napari_cool_tools_oct_preproc._oct_preproc_func import desine
from napari_cool_tools_io import device
import napari_cool_tools_io
from napari_cool_tools_registration._bidirectional_ascan_registration_funcs import (
    unwarp_polynomial_offset_torch,
    unwarp_polynomial_linear_torch,
    unwarp_polynomial_unified_torch,
    process_image_no_plot_torch,
    blur_score_vol_torch_spatial,
    blur_score_vol_torch_frequency
)

class Bidirectional_Ascan_Registration_Widget(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Bidirectional Ascan Registration Dialog")

        #initialize variables
        self.volume = None

        # Create pyqtgraph image viewer
        self.axes = {'x':1, 'y':0} #this is inverse because napari transposes images
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

        #show a random image
        self.first_time = True
        bscan = np.zeros((256, 256))
        self.viewer.setImage(bscan, autoRange=True, autoLevels = False, levels=[self.minSpinBox.value(),self.maxSpinBox.value()],
                             axes=self.axes)

        #connect signals
        self.maxSpinBox.valueChanged.connect(self.updateImage)
        self.minSpinBox.valueChanged.connect(self.updateImage)
        self.desineCheckBox.stateChanged.connect(self.updateImage)
        self.enableCheckBox.stateChanged.connect(self.updateImage)
        self.doubleSideCheckBox.stateChanged.connect(self.updateImage)
        self.flipABCheckBox.stateChanged.connect(self.updateImage)
        self.dualEdgeCheckBox.stateChanged.connect(self.updateImage)
        self.linearInterpCheckBox.stateChanged.connect(self.updateImage)
        self.cropCheckBox.stateChanged.connect(self.updateImage)
        self.inverseCheckBox.stateChanged.connect(self.updateImage)
        
        self.frameNumSpinBox.valueChanged.connect(self.updateImage)
        self.averageSpinBox.valueChanged.connect(self.updateImage)

        self.C0ScaleComboBox.setCurrentText("0.1")
        self.C1ScaleComboBox.setCurrentText("0.1")
        self.rangeSpinBox.setValue(20)

        self.C0ScaleComboBox.currentTextChanged.connect(self.updateImage)
        self.C1ScaleComboBox.currentTextChanged.connect(self.updateImage)
        self.C2ScaleComboBox.currentTextChanged.connect(self.updateImage)
        self.C3ScaleComboBox.currentTextChanged.connect(self.updateImage)
        self.C0SpinBox.valueChanged.connect(self.updateImage)
        self.C1SpinBox.valueChanged.connect(self.updateImage)
        self.C2SpinBox.valueChanged.connect(self.updateImage)
        self.C3SpinBox.valueChanged.connect(self.updateImage)
        self.autoFindPushButton.clicked.connect(self.autoFindCoeffs)
        self.splitModeComboBox.currentIndexChanged.connect(self.updateImage)

        self.updateImage()

    def updateImage(self):
        #TODO so far this function only works on normal OCT. it will not work on OCTA. I don't know why yet. I will fix it later.

        if self.volume is None:
            return

        current_idx = self.frameNumSpinBox.value()
        average_num = self.averageSpinBox.value()

        if current_idx + average_num > self.volume.shape[0]:
            average_num = 1

        new_image_ave = []

        # blur_score = 0.0

        for idx in range(average_num):

            current_idx = self.frameNumSpinBox.value() + idx
            bscan = self.volume[current_idx, :, :]

            if self.inverseCheckBox.isChecked():
                bscan = bscan[::-1,:]
          
            AA, BB = (0, 1)

            if self.flipABCheckBox.isChecked():
                AA, BB = np.flip((AA, BB))

            split = 2 #this is to handle the split mode, if split mode is 1, then we will have 4 splits, otherwise we will have 2 splits

            if self.doubleSideCheckBox.isChecked():#this means the AB is alternating
                cframe = int(np.floor(current_idx/self.bmscanSpinBox.value())) #this will handle bmscan

                if (cframe % 2):
                    AA, BB = np.flip((AA, BB))

            # print(f"Split mode: {self.splitModeComboBox.currentIndex()}")
            # if self.splitModeComboBox.currentIndex() == 1:
            #     AA, BB = 2*AA, 2*BB
            #     split = 4

            new_image_torch = torch.from_numpy(bscan.copy()).to(device=device)

            if self.linearInterpCheckBox.isChecked():
                mode = "bilinear"
            else:
                mode = "nearest"

            if self.enableCheckBox.isChecked():
                coeffs = [self.C0SpinBox.value(), self.C1SpinBox.value(), self.C2SpinBox.value(), self.C3SpinBox.value()]
                scales = [float(self.C0ScaleComboBox.currentText()), float(self.C1ScaleComboBox.currentText()),
                        float(self.C2ScaleComboBox.currentText()), float(self.C3ScaleComboBox.currentText())]
                
                coeffs = torch.as_tensor(coeffs, dtype=torch.float64, device=device)
                scales = torch.as_tensor(scales, dtype=torch.float64, device=device)

                new_image_1 = new_image_torch[:,AA::split]
                new_image_1 = unwarp_polynomial_offset_torch(new_image_1, coeffs, scales, mode=mode)
                new_image_1 = unwarp_polynomial_linear_torch(new_image_1, coeffs, scales, mode=mode)
                new_image_1 = unwarp_polynomial_unified_torch(new_image_1, coeffs, scales, mode=mode)
                new_image_torch[:,AA::split] = new_image_1

                # if self.splitModeComboBox.currentIndex() == 1:
                #     new_image_1 = new_image_torch[:,AA+1::split]
                #     new_image_1 = unwarp_polynomial_offset_torch(new_image_1, coeffs, scales, mode=mode)
                #     new_image_1 = unwarp_polynomial_linear_torch(new_image_1, coeffs, scales, mode=mode)
                #     new_image_1 = unwarp_polynomial_unified_torch(new_image_1, coeffs, scales, mode=mode)
                #     new_image_torch[:,AA+1::split] = new_image_1

                if self.dualEdgeCheckBox.isChecked():
                    new_image_2 = new_image_torch[:,BB::split]
                    new_image_2 = unwarp_polynomial_offset_torch(new_image_2, -1.0*coeffs, scales, mode=mode)
                    new_image_2 = unwarp_polynomial_linear_torch(new_image_2, -1.0*coeffs, scales,mode=mode)
                    new_image_2 = unwarp_polynomial_unified_torch(new_image_2, -1.0*coeffs,scales, mode=mode)
                    new_image_torch[:,BB::split] = new_image_2

                    # if self.splitModeComboBox.currentIndex() == 1:
                    #     new_image_2 = new_image_torch[:,BB+1::split]
                    #     new_image_2 = unwarp_polynomial_offset_torch(new_image_2, -1.0*coeffs, scales, mode=mode)
                    #     new_image_2 = unwarp_polynomial_linear_torch(new_image_2, -1.0*coeffs, scales,mode=mode)
                    #     new_image_2 = unwarp_polynomial_unified_torch(new_image_2, -1.0*coeffs,scales, mode=mode)
                    #     new_image_torch[:,BB+1::split] = new_image_2

            if self.desineCheckBox.isChecked():
                new_image_torch = desine(new_image_torch, transpose=False, scale_fac=1)

            new_image = new_image_torch.cpu().numpy()

            if self.inverseCheckBox.isChecked():
                new_image = new_image[::-1,:]

            new_image_ave.append(new_image)

        new_image_ave = np.array(new_image_ave)
        new_image = np.mean(new_image_ave,axis=0)

        if self.first_time:
            self.viewer.setImage(new_image, autoRange=True, autoLevels = False, levels=[self.minSpinBox.value(),self.maxSpinBox.value()],
                                axes=self.axes)
            self.first_time = False
        else:
            self.viewer.setImage(new_image, autoRange=False, autoLevels = False, levels=[self.minSpinBox.value(),self.maxSpinBox.value()], axes=self.axes)

    def get_output_volume(self):
        if self.volume is None:
            return

        save_volume = np.zeros(self.volume.shape, dtype=self.volume.dtype)

        if self.linearInterpCheckBox.isChecked():
            mode = "bilinear"
        else:
            mode = "nearest"

        coeffs = [self.C0SpinBox.value(), self.C1SpinBox.value(), self.C2SpinBox.value(), self.C3SpinBox.value()]
        scales = [float(self.C0ScaleComboBox.currentText()), float(self.C1ScaleComboBox.currentText()),
                float(self.C2ScaleComboBox.currentText()), float(self.C3ScaleComboBox.currentText())]
        
        coeffs = torch.as_tensor(coeffs, dtype=torch.float64, device=device)
        scales = torch.as_tensor(scales, dtype=torch.float64, device=device)

        for current_idx,bscan in enumerate(self.volume):

            if self.inverseCheckBox.isChecked():
                bscan = bscan[::-1,:] #(2048,800)

            # Default
            AA, BB = (0, 1)

            if self.flipABCheckBox.isChecked():
                AA, BB = np.flip((AA, BB))

            split = 2

            if self.doubleSideCheckBox.isChecked():
                cframe = int(np.floor(current_idx/self.bmscanSpinBox.value())) #this will handle bmscan
                if cframe % 2:
                    AA, BB = np.flip((AA, BB))

            # if self.splitModeComboBox.currentIndex() == 1:#TODO fix for split mode
            #     AA, BB = 2*AA, 2*BB
            #     split = 4

            new_image_torch = torch.from_numpy(bscan.copy()).to(device=device)

            if self.enableCheckBox.isChecked():

                new_image_1 = new_image_torch[:,AA::split]
                new_image_1 = unwarp_polynomial_offset_torch(new_image_1, coeffs, scales, mode=mode)
                new_image_1 = unwarp_polynomial_linear_torch(new_image_1, coeffs, scales, mode=mode)
                new_image_1 = unwarp_polynomial_unified_torch(new_image_1, coeffs, scales, mode=mode)
                new_image_torch[:,AA::split] = new_image_1

                # if self.splitModeComboBox.currentIndex() == 1:
                #     new_image_1 = new_image_torch[:,AA+1::split]
                #     new_image_1 = unwarp_polynomial_offset_torch(new_image_1, coeffs, scales, mode=mode)
                #     new_image_1 = unwarp_polynomial_linear_torch(new_image_1, coeffs, scales, mode=mode)
                #     new_image_1 = unwarp_polynomial_unified_torch(new_image_1, coeffs, scales, mode=mode)
                #     new_image_torch[:,AA+1::split] = new_image_1

                if self.dualEdgeCheckBox.isChecked():
                    new_image_2 = new_image_torch[:,BB::split]
                    new_image_2 = unwarp_polynomial_offset_torch(new_image_2, -1.0*coeffs, scales, mode=mode)
                    new_image_2 = unwarp_polynomial_linear_torch(new_image_2, -1.0*coeffs, scales,mode=mode)
                    new_image_2 = unwarp_polynomial_unified_torch(new_image_2, -1.0*coeffs,scales, mode=mode)
                    new_image_torch[:,BB::split] = new_image_2

                    # if self.splitModeComboBox.currentIndex() == 1:
                    #     new_image_2 = new_image_torch[:,BB+1::split]
                    #     new_image_2 = unwarp_polynomial_offset_torch(new_image_2, -1.0*coeffs, scales, mode=mode)
                    #     new_image_2 = unwarp_polynomial_linear_torch(new_image_2, -1.0*coeffs, scales,mode=mode)
                    #     new_image_2 = unwarp_polynomial_unified_torch(new_image_2, -1.0*coeffs,scales, mode=mode)
                    #     new_image_torch[:,BB+1::split] = new_image_2

            if self.desineCheckBox.isChecked():
                new_image_torch = desine(new_image_torch,transpose=False)

            new_image = new_image_torch.cpu().numpy()

            if self.inverseCheckBox.isChecked():
                new_image = new_image[::-1,:]

            save_volume[current_idx] = new_image
        
        return save_volume

    def set_input_volume(self, volume: np.ndarray):

        self.volume = volume
        d, w, h = self.volume.shape
        
        mid_index = d // 2

        self.frameNumSpinBox.setValue(mid_index)
        self.frameNumSpinBox.setMaximum(d-1)

        self.volumeShapeLabel.setText(f"Size: {d} x {w} x {h}")

        vmin, vmax = np.percentile(self.volume[mid_index], (1, 99))
        self.minSpinBox.setValue(float(vmin))
        self.maxSpinBox.setValue(float(vmax))

        
    def autoFindCoeffs(self):
        #This is now support BMSCAN, but the user need to decide manually wether the image is double side or not.
        #This will only works on VISTA BMScan!!!!!!!!!!!
        #if this is non vista, it will be considered normal OCT. not OCTA!!!

        if self.volume is None:
            return

        idx1 = self.frameNumSpinBox.value()
        idx2 = idx1 + 1 #the next frame

        if idx2 >= self.volume.shape[0]:
            idx1 = idx1 - 1
            idx2 = idx1 + 1

        bscan1 = self.volume[idx1, :, :]
        bscan2 = self.volume[idx2, :, :]

        if self.inverseCheckBox.isChecked():
            bscan1 = bscan1[::-1,:]
            bscan2 = bscan2[::-1,:]

        image1 = torch.from_numpy(bscan1.copy()).to(device=napari_cool_tools_io.device)
        image2 = torch.from_numpy(bscan2.copy()).to(device=napari_cool_tools_io.device)

        dtype = image1.dtype
        device = image1.device

        ranges = self.rangeSpinBox.value()

        c0_range = [0.0]
        c1_range = [0.0]
        c2_range = [0.0]
        c3_range = [0.0]

        c0_current = 0.0
        c1_current = 0.0
        c2_current = 0.0
        c3_current = 0.0

        if self.currentCheckBox.isChecked():
            c0_current = self.C0SpinBox.value()
            c1_current = self.C1SpinBox.value()
            c2_current = self.C2SpinBox.value()
            c3_current = self.C3SpinBox.value()         

        print(f"Starting from current coeffs: {c0_current}, {c1_current}, {c2_current}, {c3_current}")

        step_size = float(self.stepSizeComboBox.currentText())

        if self.C0CheckBox.isChecked():
            c0_range = np.arange(-ranges, ranges, 1, dtype=np.float32)*step_size + c0_current

        if self.C1CheckBox.isChecked():
            c1_range = np.arange(-ranges, ranges, 1, dtype=np.float32)*step_size + c1_current

        if self.C2CheckBox.isChecked():
            c2_range = np.arange(-ranges, ranges, 1, dtype=np.float32)*step_size + c2_current

        if self.C3CheckBox.isChecked():
            c3_range = np.arange(-ranges, ranges, 1, dtype=np.float32)*step_size + c3_current

        total_iterations = len(c0_range) * len(c1_range) * len(c2_range) * len(c3_range)

        best_score = 0
        best_coeffs = torch.as_tensor([c0_current, c1_current, c2_current, c3_current], dtype=dtype, device=device)

        sc0 = float(self.C0ScaleComboBox.currentText())
        sc1 = float(self.C1ScaleComboBox.currentText())
        sc2 = float(self.C2ScaleComboBox.currentText())
        sc3 = float(self.C3ScaleComboBox.currentText())

        scales = torch.as_tensor([sc0, sc1, sc2, sc3], dtype=dtype, device=device)
        if self.linearInterpCheckBox.isChecked():
            mode = "bilinear"
        else:
            mode = "nearest"

        AA, BB = (0, 1)

        if self.flipABCheckBox.isChecked():
            AA, BB = np.flip((AA, BB))


        AA1, BB1 = (AA, BB)
        AA2, BB2 = (AA, BB)

        if self.doubleSideCheckBox.isChecked():#this means the AB is alternating
            #if the frame is odd, flip (if they are in the same group, they may be both flipped)
            cframe1 = int(np.floor(idx1/self.bmscanSpinBox.value())) #this will handle bmscan
            if cframe1 % 2:
                AA1, BB1 = np.flip((AA, BB))
            cframe2 = int(np.floor(idx2/self.bmscanSpinBox.value())) #this will handle bmscan
            if cframe2 % 2:
                AA2, BB2 = np.flip((AA, BB))

        # initialize score #no double side (but inlcude the flip if needed)
        new_image_torch1 = process_image_no_plot_torch(image1,best_coeffs,scales,AA=AA1,BB=BB1,dual_edge=self.dualEdgeCheckBox.isChecked(),mode=mode)
        new_image_torch2 = process_image_no_plot_torch(image2,best_coeffs,scales,AA=AA2,BB=BB2,dual_edge=self.dualEdgeCheckBox.isChecked(),mode=mode)

        if self.frequencyDomainCheckBox.isChecked():
            blur_score_vol = blur_score_vol_torch_frequency
        else:
            blur_score_vol = blur_score_vol_torch_spatial

        best_score = blur_score_vol(new_image_torch1).item() + blur_score_vol(new_image_torch2).item()

        iteration = 0
        with tqdm(total=total_iterations, desc="Searching coeffs") as pbar:
            pbar.set_postfix(best_score=best_score,coeffs=best_coeffs)

            for c0 in c0_range:
                for c1 in c1_range:
                    for c2 in c2_range:
                        for c3 in c3_range:
                            iteration += 1
                            coeffs = torch.as_tensor([c0, c1, c2, c3], dtype=dtype, device=device)

                            new_image_torch1 = process_image_no_plot_torch(image1,coeffs,scales,AA=AA1,BB=BB1,dual_edge=self.dualEdgeCheckBox.isChecked(),mode=mode)
                            new_image_torch2 = process_image_no_plot_torch(image2,coeffs,scales,AA=AA2,BB=BB2,dual_edge=self.dualEdgeCheckBox.isChecked(),mode=mode)

                            score = blur_score_vol(new_image_torch1).item() + blur_score_vol(new_image_torch2).item()
                            if score < best_score:
                                best_score = score
                                best_coeffs =torch.as_tensor([c0, c1, c2, c3], dtype=dtype, device=device)
                                pbar.set_postfix(
                                    best_score=best_score,
                                    coeffs=best_coeffs.cpu().numpy()
                                )

                            pbar.update(1)

        best_coeffs = best_coeffs.cpu().numpy()

        print(f"Best coeffs found: {best_coeffs} with score: {best_score}")

        self.C0SpinBox.setValue(best_coeffs[0])
        self.C1SpinBox.setValue(best_coeffs[1])
        self.C2SpinBox.setValue(best_coeffs[2])
        self.C3SpinBox.setValue(best_coeffs[3])
        self.updateImage()
