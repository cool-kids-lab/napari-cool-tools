from qtpy.QtWidgets import QWidget, QDialog
from qtpy import QtWidgets
from napari_cool_tools_io._unp_preview import Ui_Dialog
import pyqtgraph as pg
import numpy as np
from pathlib import Path
from napari_cool_tools_io import getWindow, unp_meta
import torch
import math
from napari_cool_tools_io import device
from napari_cool_tools_io.process_unp import unpack12_torch, dc_subtraction_double_sweep_torch
from napari_cool_tools_io.process_unp import comp_dis_phase_torch, reshuffle_vista_indices
from napari_cool_tools_oct_preproc._oct_preproc_func import desine
from tqdm import tqdm

class Unp_Preview_Widget(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("UNP Preview Dialog")
        self.dcSubtractCheckBox.setChecked(True)
        self.doubleSideCheckBox.setChecked(True)

        self.OCTACheckBox.setEnabled(False)
        self.OCTAComboBox.setEnabled(False)
        self.structureCheckBox.setEnabled(False)
        self.OCTACheckBox.hide()
        self.OCTAComboBox.hide()
        self.structureCheckBox.hide()

        #class attributes
        self.meta = None
        self.unp_file_path = None
        self.raw_data = None
        self.imageIndexing = None
        # self.indices = None

        # Create pyqtgraph image viewer
        self.axes = {'x':0, 'y':1}
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

        self.dispC2BSpinBox.hide()
        self.dispC3BSpinBox.hide()

        self.splitDispersionCheckBox.stateChanged.connect(self.dispC2BSpinBox.setVisible)
        self.splitDispersionCheckBox.stateChanged.connect(self.dispC3BSpinBox.setVisible)

        self.minIntensitySpinBox.valueChanged.connect(self.updateImage)
        self.maxIntensitySpinBox.valueChanged.connect(self.updateImage)

        self.dispC2ASpinBox.valueChanged.connect(self.updateImage)
        self.dispC3ASpinBox.valueChanged.connect(self.updateImage)
        self.dispC2BSpinBox.valueChanged.connect(self.updateImage)
        self.dispC3BSpinBox.valueChanged.connect(self.updateImage)
        self.splitDispersionCheckBox.stateChanged.connect(self.updateImage)
        self.dispersionModeComboBox.currentIndexChanged.connect(self.updateImage)
        self.windowComboBox.currentIndexChanged.connect(self.updateImage)
        self.splitSpectrumCheckBox.stateChanged.connect(self.updateImage)

        self.fullRangeCheckBox.stateChanged.connect(self.updateImage)
        self.desineCheckBox.stateChanged.connect(self.updateImage)
        self.logScaleCheckBox.stateChanged.connect(self.updateImage)
        self.doubleSideCheckBox.stateChanged.connect(self.updateImage)
        self.dcSubtractCheckBox.stateChanged.connect(self.updateImage)
        self.autoCompensateButton.clicked.connect(self.autoDispersionFinder)

        self.OCTACheckBox.stateChanged.connect(self.OCTAComboBox.setEnabled)
        self.OCTACheckBox.stateChanged.connect(self.structureCheckBox.setEnabled)

        #show a random image
        bscan = np.zeros((256, 256))
        self.viewer.setImage(bscan, autoRange=True, autoLevels = False,
                              levels=[self.minIntensitySpinBox.value(),self.maxIntensitySpinBox.value()],axes=self.axes)

    def updateImage(self, autoLevels=False):

        temp_frames = []

        current_frame = self.frameNumberSpinBox.value()
        ave_number = self.frameAverageSpinBox.value() - 1

        if (current_frame+ave_number) > (self.meta.depth-1):# 799 + 2 -1> 799 (this is ok)
            current_frame = current_frame - ave_number # 799 -2 +1 = 798

        for idx in range(0, self.frameAverageSpinBox.value()):

            # Subtract the DC signal
            if self.dcSubtractCheckBox.isChecked():
                subtracted_signal = dc_subtraction_double_sweep_torch(self.raw_data[idx])
            else:
                subtracted_signal = self.raw_data[idx]

            hamming = getWindow(self.meta.width, self.windowComboBox.currentIndex(), subtracted_signal.dtype, subtracted_signal.device)

            # Hamming windowing
            hamming_signal = subtracted_signal * hamming

            img_disp_comp = torch.zeros_like(hamming_signal,dtype=torch.complex64)


            #apply dispersion first before splitting
            if self.splitDispersionCheckBox.isChecked():
                dispCoeffsA = [self.dispC2ASpinBox.value(), self.dispC3ASpinBox.value()]
                img_disp_comp[0::2] = comp_dis_phase_torch(hamming_signal[0::2], 3, dispCoeffsA, mode=self.dispersionModeComboBox.currentIndex())
                dispCoeffsB = [self.dispC2BSpinBox.value(), self.dispC3BSpinBox.value()]
                img_disp_comp[1::2] = comp_dis_phase_torch(hamming_signal[1::2], 3, dispCoeffsB, mode=self.dispersionModeComboBox.currentIndex())

            else:
                dispCoeffsA = [self.dispC2ASpinBox.value(), self.dispC3ASpinBox.value()]
                img_disp_comp = comp_dis_phase_torch(hamming_signal, 3, dispCoeffsA, mode=self.dispersionModeComboBox.currentIndex())

            # Fourier Transform
            if self.splitSpectrumCheckBox.isChecked():
                # Split Spectrum Fourier Transform
                half_point = img_disp_comp.shape[-1] // 2
                img_disp_comp_split = torch.zeros((img_disp_comp.shape[0]*2, half_point), dtype=img_disp_comp.dtype)

                img_disp_comp_split[0::4, :] = img_disp_comp[0::2, :half_point]
                img_disp_comp_split[1::4, :] = img_disp_comp[0::2, half_point:]

                img_disp_comp_split[3::4, :] = img_disp_comp[1::2, half_point:]
                img_disp_comp_split[2::4, :] = img_disp_comp[1::2, :half_point]

                fft_signal = torch.fft.ifft(img_disp_comp_split, dim=-1)

            else:
                # Standard Fourier Transform
                # img_disp_comp[1::2, :] = torch.flip(img_disp_comp[1::2, :], dims=[1]) #flip the odd frames
                fft_signal = torch.fft.ifft(img_disp_comp, dim=-1)

            if self.fullRangeCheckBox.isChecked():
                temp_frame = torch.abs(fft_signal) #full range
            else:
                temp_frame = torch.abs(fft_signal[:, int(fft_signal.shape[1] / 2):]) #take the negative part
                # temp_frame = torch.abs(fft_signal[:, :int(fft_signal.shape[1] / 2)]) #take the positive part

            if self.logScaleCheckBox.isChecked():
                temp_frame = 20 * torch.log10(temp_frame + 1e-6)

            #this is just for preview, the one in the napari will be different
            if self.doubleSideCheckBox.isChecked():
                #handle bmscan
                cframe = int(np.floor((current_frame+idx)/(self.meta.bmscan)))
                if (cframe % 2):
                    temp_frame = torch.flip(temp_frame, dims=[0])
 
            temp_frames.append(temp_frame)

        temp_frames = torch.stack(temp_frames, dim=0)
        temp_frame = torch.mean(temp_frames, dim=0)

        if self.desineCheckBox.isChecked():
            temp_frame = desine(temp_frame, mode="bilinear", transpose=True, scale_fac=2)
        
        temp_frame = temp_frame.cpu().numpy()

        # temp_frame = temp_frame[::2,:]#reverse the x axis for better visualization

        current_size = self.viewer.getImageItem().image.shape # type: ignore
        next_size = temp_frame.shape # type: ignore

        if current_size == next_size:
            self.viewer.setImage(temp_frame, autoRange=False, autoLevels = autoLevels, 
                                 levels=[self.minIntensitySpinBox.value(),self.maxIntensitySpinBox.value()], axes=self.axes)
        else:
            self.viewer.setImage(temp_frame, autoRange=True, autoLevels = autoLevels, 
                                 levels=[self.minIntensitySpinBox.value(),self.maxIntensitySpinBox.value()], axes=self.axes)

    def set_average_frames(self, average_frames:int):
        self.set_frame(self.frameNumberSpinBox.value())

    def set_frame(self, frame_number:int):

        self.raw_data = []

        ave_number = self.frameAverageSpinBox.value() - 1

        if (frame_number+ave_number) > (self.meta.depth-1):# 799 + 2 -1> 799 (this is ok)
            frame_number = frame_number - ave_number # 799 -2 +1 = 798

        indices = self.imageIndexing[frame_number:frame_number + ave_number + 1]
        # print(self.indices)
            # read 2 bytes size for uint16
        if self.meta.packed:
            data_size_bytes = int(1.5 * self.meta.width * self.meta.height)
        else:
            data_size_bytes = 2 * self.meta.width * self.meta.height

        # open file
        with open(self.unp_file_path, "rb", buffering=0) as byte_reader:
            for idx in indices:
                # Set reference A-scan to find the dispersion coefficients
                # Use center frame (b-scan) of the volume
                reference_frame = idx

                # move to center frame in binary file
                byte_reader.seek(int(data_size_bytes) * int(reference_frame), 0)

                if self.meta.packed:
                    ref_RawData = byte_reader.read(data_size_bytes)
                    array = np.frombuffer(ref_RawData, dtype="<u1")

                    if array.size != data_size_bytes:
                        array = np.zeros(data_size_bytes, dtype="<u1")

                    array = torch.tensor(array).to(device)
                    array = unpack12_torch(array)
                    array = array.reshape((self.meta.height, self.meta.width))
                else:    
                    ref_RawData = byte_reader.read(data_size_bytes)
                    array = np.frombuffer(ref_RawData, dtype=np.uint16)

                    if array.size != self.meta.width * self.meta.height:
                        array = np.zeros(self.meta.height*self.meta.width, dtype=np.uint16)

                    array = array.reshape((self.meta.height, self.meta.width)).astype(np.float32)
                    array = torch.tensor(array).to(device)

                self.raw_data.append(array)

        self.updateImage()

    def set_unp_path(self, unp_file_path:Path, meta: unp_meta):
        self.unp_file_path = unp_file_path
        self.meta = meta

        #update the fileinfo label
        self.fileNameValue.setText(unp_file_path.name)
        self.imageWidthValue.setText(str(meta.width))
        self.imageHeightValue.setText(str(meta.height))
        self.imageTotalFramesValue.setText(str(meta.depth))
        self.imageBMScanValue.setText(str(meta.bmscan))


        self.imageIndexing = np.arange(meta.depth)#start from 0

        if meta.bmscan > 1:
            self.OCTACheckBox.show()
            self.OCTAComboBox.show()
            self.structureCheckBox.show()
            self.OCTACheckBox.setEnabled(True)

            if meta.vista > 1:
                self.imageIndexing = reshuffle_vista_indices(self.imageIndexing, meta.vista, meta.bmscan)

        #TODO Hande the sine puase scanning properly
        if meta.pattern == "Sine_Pause":
            #remove pause indices from the indexing
            start_pause_indices = meta.sine_frame_indices[0::2]
            stop_pause_indices = meta.sine_frame_indices[1::2]
            pause_indices = []
            for idx in range(len(start_pause_indices)):
                pause_indices.append(np.arange(start_pause_indices[idx], stop_pause_indices[idx]))

            pause_indices = np.concatenate(pause_indices)
            self.imageIndexing = self.imageIndexing[~np.isin(self.imageIndexing, pause_indices)]

        self.frameNumberSpinBox.setMaximum(len(self.imageIndexing)-1)#start from 0
        self.frameNumberSpinBox.setValue(math.ceil(len(self.imageIndexing) / 2))#e.g 420 (with 0 based index)
        self.set_frame(math.ceil(len(self.imageIndexing) / 2))

        current_image  = self.viewer.getImageItem().image

        vmin, vmax = np.percentile(current_image, (1, 99))
        self.viewer.setLevels(float(vmin), float(vmax))
        self.maxIntensitySpinBox.setValue(vmax)
        self.minIntensitySpinBox.setValue(vmin)

        self.frameNumberSpinBox.valueChanged.connect(self.set_frame)
        self.frameAverageSpinBox.valueChanged.connect(self.set_average_frames)

        self.autoDispersionFinder()
    

    def autoDispersionFinder(self):

        #first get the raw data
        temp_raw_data = []

        for idx in range(0, self.frameAverageSpinBox.value()):
            # Subtract the DC signal
            if self.dcSubtractCheckBox.isChecked():
                subtracted_signal = dc_subtraction_double_sweep_torch(self.raw_data[idx])
            else:
                subtracted_signal = self.raw_data[idx]

            # hamming = torch.hamming_window(self.meta.width, periodic=False, dtype=subtracted_signal.dtype, device=subtracted_signal.device)
            hamming = getWindow(self.meta.width, self.windowComboBox.currentIndex(), subtracted_signal.dtype, subtracted_signal.device)

            # Hamming windowing
            hamming_signal = subtracted_signal * hamming

            temp_raw_data.append(hamming_signal)

        temp_raw_data = torch.stack(temp_raw_data, dim=0)

        print(f'temp_raw_data.shape: {temp_raw_data.shape}')

        maxDispOrders = 3
        coefRange = self.autoDispRangeSpinBox.value()

        if self.splitDispersionCheckBox.isChecked():
            #dispersion for the even frames
            arrCountDispCoeffA = [0,0]

            for idx_CounterDispCoef in tqdm(
                range(0, len(arrCountDispCoeffA)), desc="Calculating Dispersion Coefficients A"
            ):
                arrDispCoeffRange = np.arange(-1 * coefRange, coefRange + 1, 1)
                arrCost = np.zeros((arrDispCoeffRange.shape[0]))

                for k in tqdm(range(0, len(arrDispCoeffRange)), desc="Calculating Costs A"):
                    arrCountDispCoeffA[idx_CounterDispCoef] = arrDispCoeffRange[k]
                    arrCost[k] = self.cal_cost_function_torch(temp_raw_data[:,0::2,:], maxDispOrders, arrCountDispCoeffA, dispersion_mode=self.dispersionModeComboBox.currentIndex())
                    # print(f'k={k}, cost={arrCost[k]}')

                argMinCost = arrCost.argmax()
                arrCountDispCoeffA[idx_CounterDispCoef] = arrDispCoeffRange[argMinCost]

            arrCountDispCoeffA = arrCountDispCoeffA

            self.dispC2ASpinBox.setValue(float(arrCountDispCoeffA[0]))
            self.dispC3ASpinBox.setValue(float(arrCountDispCoeffA[1]))

            #dispersion for the odd frames
            arrCountDispCoeffB = [0,0]

            for idx_CounterDispCoef in tqdm(
                range(0, len(arrCountDispCoeffB)), desc="Calculating Dispersion Coefficients B"
            ):
                arrDispCoeffRange = np.arange(-1 * coefRange, coefRange + 1, 1)
                arrCost = np.zeros((arrDispCoeffRange.shape[0]))

                for k in tqdm(range(0, len(arrDispCoeffRange)), desc="Calculating Costs B"):
                    arrCountDispCoeffB[idx_CounterDispCoef] = arrDispCoeffRange[k]
                    arrCost[k] = self.cal_cost_function_torch(temp_raw_data[:,1::2,:], maxDispOrders, arrCountDispCoeffB, dispersion_mode=self.dispersionModeComboBox.currentIndex())
                    # print(f'k={k}, cost={arrCost[k]}')

                argMinCost = arrCost.argmax()
                arrCountDispCoeffB[idx_CounterDispCoef] = arrDispCoeffRange[argMinCost]

            arrCountDispCoeffB = arrCountDispCoeffB

            self.dispC2BSpinBox.setValue(float(arrCountDispCoeffB[0]))
            self.dispC3BSpinBox.setValue(float(arrCountDispCoeffB[1]))

        else:

            arrCountDispCoeff = [0,0]

            for idx_CounterDispCoef in tqdm(
                range(0, len(arrCountDispCoeff)), desc="Calculating Dispersion Coefficients"
            ):
                arrDispCoeffRange = np.arange(-1 * coefRange, coefRange + 1, 1)
                arrCost = np.zeros((arrDispCoeffRange.shape[0]))

                for k in tqdm(range(0, len(arrDispCoeffRange)), desc="Calculating Costs"):
                    arrCountDispCoeff[idx_CounterDispCoef] = arrDispCoeffRange[k]
                    arrCost[k] = self.cal_cost_function_torch(temp_raw_data, maxDispOrders, arrCountDispCoeff, dispersion_mode=self.dispersionModeComboBox.currentIndex())

                argMinCost = arrCost.argmax()
                arrCountDispCoeff[idx_CounterDispCoef] = arrDispCoeffRange[argMinCost]

            arrCountDispCoeff = arrCountDispCoeff

            self.dispC2ASpinBox.setValue(float(arrCountDispCoeff[0]))
            self.dispC3ASpinBox.setValue(float(arrCountDispCoeff[1]))


    def cal_cost_function_torch(self, raw_data_list : torch.tensor, dispMaxOrder, dispCoeffs: list, dispersion_mode=0):

        temp_frames = []

        
        current_frame_num = self.frameNumberSpinBox.value()
        cframe = int(np.floor((current_frame_num)/(self.meta.bmscan)))

        for idx in range(0, len(raw_data_list)):
            data_disp_comp = comp_dis_phase_torch(raw_data_list[idx], dispMaxOrder, dispCoeffs, mode=dispersion_mode)

            # FFT magnitude squared
            toct = torch.abs(torch.fft.ifft(data_disp_comp, dim=-1))
            
            # Avoid edges
            temp_frame = toct[:, int(data_disp_comp.shape[1] / 2) + 50 : -50] #take the negative part

            #determined double side
            if self.doubleSideCheckBox.isChecked():
                if (cframe % 2):
                    temp_frame = torch.flip(temp_frame, dims=[0])

            temp_frames.append(temp_frame)

        temp_frames = torch.stack(temp_frames, dim=0)

        # #determined double side
        # if self.doubleSideCheckBox.isChecked():
        #     current_frame_num = self.frameNumberSpinBox.value()
        #     cframe = int(np.floor((current_frame_num)/(self.meta.bmscan)))
        #     if (cframe % 2):
        #         temp_frames[1::2] = torch.flip(temp_frames[1::2], dims=[1])

        # if self.doubleSideCheckBox.isChecked():
        #     temp_frames[1::2] = torch.flip(temp_frames[1::2], dims=[1])

        mean_frame = torch.mean(temp_frames, dim=0)**2

        # Normalize
        norm_oct = mean_frame / torch.sum(mean_frame)
        
        # Shannon entropy
        eps = 1e-12#this is to avoid nan
        entropy = norm_oct * torch.log10(norm_oct + eps)
        
        # Final cost
        cost = torch.sum(entropy)
        return cost