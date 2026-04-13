import xml.etree.ElementTree as ET
from qtpy.QtWidgets import QDialog
from qtpy import QtWidgets
from split_spectrum_form import Ui_Dialog
import pyqtgraph as pg
import numpy as np
from napari_cool_tools_io import unp_meta
from pathlib import Path
import os.path as ospath
import configparser
import math
import torch
from napari_cool_tools_io.process_unp import unpack12_torch
from napari_cool_tools_io.process_unp import comp_dis_phase_torch, reshuffle_vista_indices
from napari_cool_tools_io import device
from napari_cool_tools_oct_preproc._oct_preproc_func import desine

class Split_Spectrum_Widget(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Split Spectrum Dialog")

        # Create pyqtgraph image viewer
        self.axes = {'x':0, 'y':1}
        self.viewer = pg.ImageView(parent=self)
        self.viewer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.viewer.updateGeometry()

        layout = self.graphicsViewPlaceHolder.parent().layout()
        layout.replaceWidget(self.graphicsViewPlaceHolder, self.viewer)
        self.graphicsViewPlaceHolder.deleteLater()

        # Hide unnecessary buttons
        self.viewer.ui.roiBtn.hide()
        self.viewer.ui.menuBtn.hide()
        self.viewer.ui.histogram.hide()

        # Show random image for demonstration
        image = np.random.rand(256, 256) * 255
        self.viewer.setImage(image.astype(np.float32))

        
        # Create pyqtgraph plot viewer
        self.plotter = pg.PlotWidget()
        self.plotter.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.plotter.setBackground('w')
        self.plotter.setMouseEnabled(x=False, y=False)

        layout = self.plotViewPlaceHolder.parent().layout()
        layout.replaceWidget(self.plotViewPlaceHolder, self.plotter)
        self.plotViewPlaceHolder.deleteLater()

        # Data
        t = np.linspace(0, 2*np.pi, 1000)
        y = np.sin(3*t) * np.exp(-0.2*t)

        self.plotSignal =  self.plotter.plot(t, y, pen=pg.mkPen("b", width=2))
        self.plotWindow1 =  self.plotter.plot(t, y, pen=pg.mkPen("r", width=2))
        self.plotWindow2 =  self.plotter.plot(t, y, pen=pg.mkPen("g", width=2))
        self.plotWindow3 =  self.plotter.plot(t, y, pen=pg.mkPen("y", width=2))
        self.plotWindow3.setVisible(False)
        self.plotWindow4 =  self.plotter.plot(t, y, pen=pg.mkPen("c", width=2))
        self.plotWindow4.setVisible(False)

        self.loadFileButton.clicked.connect(self.on_loadButton_clicked)
        self.frameNumSpinBox.valueChanged.connect(self.set_frame)
        self.averageSpinBox.valueChanged.connect(lambda: self.set_frame(self.frameNumSpinBox.value()))
        self.dcSubtractCheckBox.stateChanged.connect(self.updateImage)
        self.desineCheckBox.stateChanged.connect(self.updateImage)
        self.doubleSideCheckBox.stateChanged.connect(self.updateImage)

        self.maxSpinBox.valueChanged.connect(self.updateImage)
        self.minSpinBox.valueChanged.connect(self.updateImage)

        #class attributes
        self.meta = unp_meta()
        self.raw_data = []
        self.unp_file_path = ""
        self.imageIndexing = []

    
    def on_loadButton_clicked(self):
        print("Load button clicked")

        #load the file
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "", "UNP Files (*.unp)")
        if file_path:
            print(f"Selected file: {file_path}")
            # Here you would add code to load the .unp file and update the viewer
            # For demonstration, we'll just print the file path and update the image with random data

            # read the xml or ini file information
            self.set_unp_path(file_path)

            if self.meta is None:
                print("Failed to read metadata. Cannot proceed.")
                return

            self.volumeShapeLabel.setText(f"Volume Shape: {self.meta.height} x {self.meta.width} x {self.meta.depth} x {self.meta.bmscan}")

            # auto contrast
            current_image  = self.viewer.getImageItem().image
            vmin, vmax = np.percentile(current_image, (1.0, 99.0)) # type: ignore
            self.viewer.setLevels(float(vmin), float(vmax))
            self.maxSpinBox.setValue(vmax)
            self.minSpinBox.setValue(vmin)

        else:
            print("No file selected")


    def dc_subtraction_single_sweep_torch(self,data: torch.Tensor) -> torch.Tensor:

        if data.shape[0] % 2 != 0:
            raise ValueError("Number of A-scans must be even for double-sweep subtraction")

        corrected = data.clone()
        corrected = corrected - torch.mean(corrected, dim=0, keepdim=True)

        # Recombine into full B-scan
        subtracted_signal = torch.zeros_like(data)
        subtracted_signal = corrected

        return subtracted_signal


    def updateImage(self, autoLevels=False):
        temp_frames = []

        if self.meta is None:
            print("Metadata not loaded. Cannot update image.")
            return

        current_frame = self.frameNumSpinBox.value()
        ave_number = self.averageSpinBox.value() - 1

        if (current_frame+ave_number) > (self.meta.depth-1):# 799 + 2 -1> 799 (this is ok)
            current_frame = current_frame - ave_number # 799 -2 +1 = 798

        for idx in range(0, self.averageSpinBox.value()):
            # Subtract the DC signal
            if self.dcSubtractCheckBox.isChecked():
                subtracted_signal = self.dc_subtraction_single_sweep_torch(self.raw_data[idx])
            else:
                subtracted_signal = self.raw_data[idx]

            #split signal into 2 halves
            subtracted_signal_1 = subtracted_signal[:, :self.meta.width//2]
            subtracted_signal_2 = subtracted_signal[:, self.meta.width//2:]

            subtracted_signal_2 = torch.flip(subtracted_signal_2, dims=[1])#reverse the x axis for consistent wavelength ordering

            hamming_1 = torch.hamming_window(self.meta.width//2, periodic=False, dtype=subtracted_signal.dtype, device=subtracted_signal.device)
            hamming_2 = torch.hamming_window(self.meta.width//2, periodic=False, dtype=subtracted_signal.dtype, device=subtracted_signal.device)

            # Hamming windowing
            hamming_signal_1 = subtracted_signal_1 * hamming_1
            hamming_signal_2 = subtracted_signal_2 * hamming_2

            img_disp_comp_1 = torch.zeros_like(hamming_signal_1,dtype=torch.complex64)

            dispCoeffs_1 = [0.0, 0.0]
            img_disp_comp_1 = comp_dis_phase_torch(hamming_signal_1, 3, dispCoeffs_1, mode=0)

            img_disp_comp_2 = torch.zeros_like(hamming_signal_2,dtype=torch.complex64)

            dispCoeffs_2 = [0.0, 0.0]
            img_disp_comp_2 = comp_dis_phase_torch(hamming_signal_2, 3, dispCoeffs_2, mode=0)

            # Fourier Transform
            fft_signal_1 = torch.fft.ifft(img_disp_comp_1, dim=-1)
            fft_signal_2 = torch.fft.ifft(img_disp_comp_2, dim=-1)

            temp_frame = torch.zeros((self.meta.height*2, self.meta.width//4), dtype=torch.float32, device=fft_signal_1.device)
            temp_frame[0::2] = torch.abs(fft_signal_1[:, int(fft_signal_1.shape[1] / 2):]) #take the negative part
            temp_frame[1::2] = torch.abs(fft_signal_2[:, int(fft_signal_2.shape[1] / 2):]) #take the negative part

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
                                    levels=[self.minSpinBox.value(),self.maxSpinBox.value()], axes=self.axes)
        else:
            self.viewer.setImage(temp_frame, autoRange=True, autoLevels = autoLevels, 
                                    levels=[self.minSpinBox.value(),self.maxSpinBox.value()], axes=self.axes)
            
        
        # Data
        t = np.linspace(0, self.meta.width, self.meta.width)
        y = self.raw_data[0][0,:].cpu().numpy()

        #update plot
        self.plotSignal.setData(x=t, y=y)

        
        # # N = self.meta.width          # total array length
        # W1 = self.meta.width         # width of the Gaussian segment (non-zero region)
        # peak = y.max()        # peak amplitude of the Gaussian
        # sigma = W1 / 6.0   # standard deviation in samples (~±3σ spans ~W)

        # # Build the 512-sample Gaussian centered at 0
        # x = np.arange(W1) - (W1 - 1) / 2.0   # symmetric around 0
        # g = np.exp(-0.5 * (x / sigma)**2)
        # g *= peak / g.max()                # normalize to desired peak

        # #update plot
        # self.plotWindow1.setData(x=x, y=g)

        # self.plotSignal.setVisible(False)

        return


    def unp_proc_meta(self, path:str) -> unp_meta | None:

        print(f"\nOpening file: {path}")

        head, tail = ospath.split(path)
        file_no_ext = tail.split(".")[0]

        # constuct path to metafile assumed to be in same directory
        meta_path_xml = ospath.join(head, file_no_ext + ".xml")
        meta_path_ini = ospath.join(head, file_no_ext + ".ini")

        # Initialize metadata container
        meta = unp_meta()
        #width, height, depth = [4096, 800, 840]

        if Path(meta_path_ini).is_file():
            print(".ini Meta Data exists:")

            config = configparser.ConfigParser()
            config.read(meta_path_ini)

            meta.width = config.getint('General', 'WIDTH')
            meta.height = config.getint('General', 'HEIGHT')
            meta.depth = config.getint('General', 'FRAMES')
            meta.bmscan = config.getint('OCTA', 'BMScan')
            meta.vista = config.getint('Scanning', 'VISTA_Num')
            meta.packed = config.getboolean('Acquisition', 'PACKED12')
            meta.double_side = config.getboolean('Scanning', 'Bidirectional')
            meta.pattern = config['Scanning']['Pattern']
            meta.delay = config.getint('Scanning', 'XDelay')

            if meta.pattern == "Sine_Pause":

                if config.has_option('Scanning', 'Sine_Pause_Frame_Index'):
                    meta.sine_frame_indices = list(map(int, config['Scanning']['Sine_Pause_Frame_Index'].split()))
                    meta.sine_hires_ratio = config.getint('Scanning', 'Sine_Pause_X_Rate_Reduction')
                else:
                    meta.sine_frame_indices = [236, 256, 286, 306, 336, 356, 386, 406, 434, 434]
                    meta.sine_hires_ratio = 2


            meta.width = meta.width*2
            meta.height = meta.height//2

            return meta

        if Path(meta_path_xml).is_file():
            print(".xml Meta Data exists:")

            tree = ET.parse(meta_path_xml)
            root = tree.getroot()
            volume_size = root.find(".//Volume_Size")
            volume_size_attrib = volume_size.attrib # type: ignore
            meta.height = int(volume_size_attrib["Height"])
            meta.width = int(volume_size_attrib["Width"])
            meta.depth = int(volume_size_attrib["Number_of_Frames"])

            scanning_params = root.find(".//Scanning_Parameters")
            scanning_params_attrib = scanning_params.attrib # type: ignore
            meta.bmscan = int(scanning_params_attrib["Number_of_BM_scans"])

            meta.width = meta.width*2
            meta.height = meta.height//2

            return meta
        
        # case no metadata found
        print("No metadata file found.")
        return None
    

    def set_frame(self, frame_number:int):

        if self.meta is None:
            print("Metadata not loaded. Cannot set frame.")
            return

        self.raw_data = []

        ave_number = self.averageSpinBox.value() - 1

        if (frame_number+ave_number) > (self.meta.depth-1):# 799 + 2 -1> 799 (this is ok)
            frame_number = frame_number - ave_number # 799 -2 +1 = 798

        indices = self.imageIndexing[frame_number:frame_number + ave_number + 1]

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

                    if array.size != self.meta.width * self.meta.height * 3 // 2:
                        array = np.zeros(self.meta.height*self.meta.width, dtype="<u1")

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


    def set_unp_path(self, unp_file_path:str):
        self.unp_file_path = Path(unp_file_path)
        self.meta = self.unp_proc_meta(unp_file_path)

        if self.meta is None:
            print("Failed to read metadata. Cannot proceed.")
            return

        self.imageIndexing = np.arange(self.meta.depth)#start from 0

        if self.meta.bmscan > 1:

            if self.meta.vista > 1:
                self.imageIndexing = reshuffle_vista_indices(self.imageIndexing, self.meta.vista, self.meta.bmscan)

        self.frameNumSpinBox.setMaximum(len(self.imageIndexing)-1)#start from 0
        self.frameNumSpinBox.setValue(math.ceil(len(self.imageIndexing) / 2))#e.g 420 (with 0 based index)
        self.set_frame(math.ceil(len(self.imageIndexing) / 2))