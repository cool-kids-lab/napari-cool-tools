from qtpy.QtWidgets import QDialog
from qtpy import QtWidgets
from napari_cool_tools_registration._bidirectional_ascan_registration_form import Ui_Dialog
import pyqtgraph as pg
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from napari_cool_tools_oct_preproc._oct_preproc_func import desine
from napari_cool_tools_io import device
import napari_cool_tools_io

def blur_score_vol_torch_frequency(img: torch.Tensor):
    """
    Edge magnitude via frequency-domain derivatives.
    img: (H, W) real tensor (any float dtype); returns a scalar score (sum of magnitudes).
    """
    assert img.ndim == 2, "Input must be 2D (H, W)."

    # #avoid top edges
    # img[:10,:] = 0
    # img[-10:,:] = 0

    H, W = img.shape
    device, dtype = img.device, img.dtype

    # 1) FFT (no fftshift)
    F = torch.fft.fft2(img)

    # 2) Frequency coordinates shaped (H, W)
    u = torch.fft.fftfreq(W, d=1.0, device=device, dtype=dtype)   # (W,)
    v = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype)   # (H,)
    V, U = torch.meshgrid(v, u, indexing='ij')                    # both (H, W)

    # 3) Derivative filters: j*2*pi*f
    Hx = torch.complex(torch.zeros_like(U), 2 * torch.pi * U)     # (H, W) complex
    Hy = torch.complex(torch.zeros_like(V), 2 * torch.pi * V)

    # 4) Apply filters in frequency
    Fx = F * Hx
    Fy = F * Hy

    # 5) Inverse FFT to spatial gradients
    edge_x = torch.fft.ifft2(Fx).real
    edge_y = torch.fft.ifft2(Fy).real

    # 6) Gradient magnitude + a simple score
    edge_mag = torch.hypot(edge_x, edge_y)                        # sqrt(x^2 + y^2)
    score = edge_mag.abs().sum()                                  # torch scalar

    return score

def blur_score_vol_torch_spatial(x: torch.Tensor) -> torch.Tensor:
    """
    Sum of absolute Laplacian (higher => sharper).
    Returns a *torch scalar* (zero-dim tensor) on the same device, float64.

    I: (H,W) or (D,H,W). For 3D, applies 2D Laplacian per slice.
    """
    # assert I.ndim in (2, 3), "I must be (H,W) or (D,H,W)"
    # x = I.to(torch.float64)

    # Shape to (N,1,H,W) for conv2d
    x4 = x.unsqueeze(0).unsqueeze(0) if x.ndim == 2 else x.unsqueeze(1)

    # 3x3 Laplacian kernel
    h = torch.tensor([[0., 1., 0.],
                      [1., -4., 1.],
                      [0., 1., 0.]], dtype=x.dtype, device=x.device).view(1,1,3,3)

    # replicate-pad edges (nearest) then conv
    xpad = F.pad(x4, (1, 1, 1, 1), mode='replicate')
    L = F.conv2d(xpad, h)  # (N,1,H,W)

    # Sum of absolute Laplacian -> torch scalar
    score = L.abs().sum()
    return score

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
            split = 2

            if self.flipABCheckBox.isChecked():
                if current_idx % 2:
                    AA, BB = (1, 0)

            if self.splitModeComboBox.currentIndex() == 1:
                AA, BB = 2*AA, 2*BB
                split = 4

            new_image_torch = torch.from_numpy(bscan.copy()).to(device=device)

            #TODO handle double side images
            cframe = int(np.floor(current_idx/self.bmscanSpinBox.value()))
            if (cframe % 2):
                new_image_torch = torch.flip(new_image_torch, dims=[0])

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
                new_image_1 = self.unwarp_polynomial_offset_torch(new_image_1, coeffs, scales, mode=mode)
                new_image_1 = self.unwarp_polynomial_linear_torch(new_image_1, coeffs, scales, mode=mode)
                new_image_1 = self.unwarp_polynomial_unified_torch(new_image_1, coeffs, scales, mode=mode)
                new_image_torch[:,AA::split] = new_image_1

                if self.splitModeComboBox.currentIndex() == 1:
                    new_image_1 = new_image_torch[:,AA+1::split]
                    new_image_1 = self.unwarp_polynomial_offset_torch(new_image_1, coeffs, scales, mode=mode)
                    new_image_1 = self.unwarp_polynomial_linear_torch(new_image_1, coeffs, scales, mode=mode)
                    new_image_1 = self.unwarp_polynomial_unified_torch(new_image_1, coeffs, scales, mode=mode)
                    new_image_torch[:,AA+1::split] = new_image_1

                if self.dualEdgeCheckBox.isChecked():
                    new_image_2 = new_image_torch[:,BB::split]
                    # coeffs = -1.0*coeffs
                    new_image_2 = self.unwarp_polynomial_offset_torch(new_image_2, -1.0*coeffs, scales, mode=mode)
                    new_image_2 = self.unwarp_polynomial_linear_torch(new_image_2, -1.0*coeffs, scales,mode=mode)
                    new_image_2 = self.unwarp_polynomial_unified_torch(new_image_2, -1.0*coeffs,scales, mode=mode)
                    new_image_torch[:,BB::split] = new_image_2

                    if self.splitModeComboBox.currentIndex() == 1:
                        new_image_2 = new_image_torch[:,BB+1::split]
                        # coeffs = -1.0*coeffs
                        new_image_2 = self.unwarp_polynomial_offset_torch(new_image_2, -1.0*coeffs, scales, mode=mode)
                        new_image_2 = self.unwarp_polynomial_linear_torch(new_image_2, -1.0*coeffs, scales,mode=mode)
                        new_image_2 = self.unwarp_polynomial_unified_torch(new_image_2, -1.0*coeffs,scales, mode=mode)
                        new_image_torch[:,BB+1::split] = new_image_2

            if self.desineCheckBox.isChecked():
                new_image_torch = desine(new_image_torch, transpose=False, scale_fac=1)

            #TODO handle double side images
            cframe = int(np.floor(current_idx/self.bmscanSpinBox.value()))
            if (cframe % 2):
                new_image_torch = torch.flip(new_image_torch, dims=[0])

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
            split = 2

            if self.flipABCheckBox.isChecked():
                if current_idx % 2: #TODO handle bmscan
                    AA, BB = (1, 0)

            if self.splitModeComboBox.currentIndex() == 1:
                AA, BB = 2*AA, 2*BB
                split = 4


            new_image_torch = torch.from_numpy(bscan.copy()).to(device=device)

            #TODO handle double side images
            cframe = int(np.floor(current_idx/self.bmscanSpinBox.value()))
            if (cframe % 2):
                new_image_torch = torch.flip(new_image_torch, dims=[0])

            if self.enableCheckBox.isChecked():

                new_image_1 = new_image_torch[:,AA::split]
                new_image_1 = self.unwarp_polynomial_offset_torch(new_image_1, coeffs, scales, mode=mode)
                new_image_1 = self.unwarp_polynomial_linear_torch(new_image_1, coeffs, scales, mode=mode)
                new_image_1 = self.unwarp_polynomial_unified_torch(new_image_1, coeffs, scales, mode=mode)
                new_image_torch[:,AA::split] = new_image_1

                if self.splitModeComboBox.currentIndex() == 1:
                    new_image_1 = new_image_torch[:,AA+1::split]
                    new_image_1 = self.unwarp_polynomial_offset_torch(new_image_1, coeffs, scales, mode=mode)
                    new_image_1 = self.unwarp_polynomial_linear_torch(new_image_1, coeffs, scales, mode=mode)
                    new_image_1 = self.unwarp_polynomial_unified_torch(new_image_1, coeffs, scales, mode=mode)
                    new_image_torch[:,AA+1::split] = new_image_1

                if self.dualEdgeCheckBox.isChecked():
                    new_image_2 = new_image_torch[:,BB::split]
                    # coeffs = -1.0*coeffs
                    new_image_2 = self.unwarp_polynomial_offset_torch(new_image_2, -1.0*coeffs, scales, mode=mode)
                    new_image_2 = self.unwarp_polynomial_linear_torch(new_image_2, -1.0*coeffs, scales,mode=mode)
                    new_image_2 = self.unwarp_polynomial_unified_torch(new_image_2, -1.0*coeffs,scales, mode=mode)
                    new_image_torch[:,BB::split] = new_image_2

                    if self.splitModeComboBox.currentIndex() == 1:
                        new_image_2 = new_image_torch[:,BB+1::split]
                        # coeffs = -1.0*coeffs
                        new_image_2 = self.unwarp_polynomial_offset_torch(new_image_2, -1.0*coeffs, scales, mode=mode)
                        new_image_2 = self.unwarp_polynomial_linear_torch(new_image_2, -1.0*coeffs, scales,mode=mode)
                        new_image_2 = self.unwarp_polynomial_unified_torch(new_image_2, -1.0*coeffs,scales, mode=mode)
                        new_image_torch[:,BB+1::split] = new_image_2

            if self.desineCheckBox.isChecked():
                new_image_torch = desine(new_image_torch,transpose=False)

            #TODO handle double side images
            cframe = int(np.floor(current_idx/self.bmscanSpinBox.value()))
            if (cframe % 2):
                new_image_torch = torch.flip(new_image_torch, dims=[0])

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


    def UnwarpPolynomialUnified(self, frameData: np.ndarray, coeffs, scales,
                                centered: bool = False
                                ) -> np.ndarray:
        """
        Python version of MATLAB's UnwarpPolynomialNonLinear.
        
        Parameters
        ----------
        frameData : 2D np.ndarray
            Input image [h, w]
        coeffs : list or np.ndarray
            Polynomial coefficients [c0, c1, c2, c3]
        
        Returns
        -------
        result : 2D np.ndarray
            Warped image with bilinear interpolation
        """
        h, w = frameData.shape

        # Normalized Y input from 0 to 1
        y_input = np.linspace(-0.5, 0.5, w) if centered else np.linspace(0, 1, w) # type: ignore

        sc0 = scales[0]
        sc1 = scales[1]
        sc2 = scales[2]
        sc3 = scales[3]

        # Apply offset
        offset = sc0 * coeffs[0] / w
        y_input = y_input + offset

        # Apply linear scaling
        linear_scale = (w + (coeffs[1] * sc1)) / w

        # Polynomial warp
        y_warp = (
            linear_scale * y_input
            + (coeffs[2] / sc2) * y_input * np.abs(y_input)
            + (coeffs[3] / sc3) * y_input**3
        )

        # Normalize back to 0..1
        y_warp_norm = (y_warp - y_warp.min()) / (y_warp.max() - y_warp.min())

        # Map to pixel indices
        y_idx = y_warp_norm * (w - 1)

        # --- Interpolation (bilinear like MATLAB's interp2) ---
        xx = np.arange(h) #840
        yy = np.arange(w) #800
        interpolator = RegularGridInterpolator(
            (xx, yy), frameData, method="linear", bounds_error=False, fill_value=0
        )

        x_grid, y_grid = np.meshgrid(xx, y_idx, indexing='ij')

        # Points to sample (N, 2)
        pts = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        result = interpolator(pts).reshape(h, w)

        return result
    

    def autoFindCoeffs(self):

        if self.volume is None:
            return

        idx1 = self.frameNumSpinBox.value()
        idx2 = idx1 + 1

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

        flipAB = [0,1]

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

        total_iterations = (
            len(c0_range) * len(c1_range) * len(c2_range) * len(c3_range) * len(flipAB)
        )

        best_score = 0
        best_coeffs = torch.as_tensor([c0_current, c1_current, c2_current, c3_current], dtype=dtype, device=device)
        best_flip = 0

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
        # initialize score
        new_image_torch1 = self.processImageNoPlot_torch(image1,best_coeffs,scales,AA=AA,BB=BB,mode=mode)
        new_image_torch2 = self.processImageNoPlot_torch(image2,best_coeffs,scales,AA=AA,BB=BB,mode=mode)

        if self.frequencyDomainCheckBox.isChecked():
            blur_score_vol = blur_score_vol_torch_frequency
        else:
            blur_score_vol = blur_score_vol_torch_spatial

        best_score = blur_score_vol(new_image_torch1).item() + blur_score_vol(new_image_torch2).item()

        iteration = 0
        with tqdm(total=total_iterations, desc="Searching coeffs") as pbar:
            pbar.set_postfix(best_score=best_score,coeffs=best_coeffs, flip_status=0)

            for flip in flipAB:
                if flip:

                    if idx1 % 2: #TODO handle bmscan
                        AA, BB = (1, 0)
                
                    for c0 in c0_range:
                        for c1 in c1_range:
                            for c2 in c2_range:
                                for c3 in c3_range:
                                    iteration += 1
                                    coeffs = torch.as_tensor([c0, c1, c2, c3], dtype=dtype, device=device)

                                    new_image_torch1 = self.processImageNoPlot_torch(image1,coeffs,scales,AA=AA,BB=BB,mode=mode)
                                    new_image_torch2 = self.processImageNoPlot_torch(image2,coeffs,scales,AA=BB,BB=AA,mode=mode)

                                    score = blur_score_vol(new_image_torch1).item() + blur_score_vol(new_image_torch2).item()
                                    if score < best_score:
                                        best_score = score
                                        best_coeffs = [c0, c1, c2, c3]
                                        best_flip = flip
                                        pbar.set_postfix(
                                            best_score=best_score,
                                            coeffs=best_coeffs, flip_status=best_flip
                                        )

                                    pbar.update(1)
                
                else:
                    
                    AA, BB = (0, 1)
                    
                    for c0 in c0_range:
                        for c1 in c1_range:
                            for c2 in c2_range:
                                for c3 in c3_range:
                                    iteration += 1
                                    coeffs = torch.as_tensor([c0, c1, c2, c3], dtype=dtype, device=device)
                                    
                                    new_image_torch1 = self.processImageNoPlot_torch(image1,coeffs,scales,AA=AA,BB=BB,mode=mode)
                                    new_image_torch2 = self.processImageNoPlot_torch(image2,coeffs,scales,AA=AA,BB=BB,mode=mode)
                                    score = blur_score_vol(new_image_torch1).item() + blur_score_vol(new_image_torch2).item()
                                    if score < best_score:
                                        best_score = score
                                        best_coeffs = [c0, c1, c2, c3]
                                        best_flip = flip
                                        pbar.set_postfix(
                                            best_score=best_score,
                                            coeffs=best_coeffs, flip_status=best_flip
                                        )

                                    pbar.update(1)

        print(f"Best coeffs found: {best_coeffs} with score: {best_score}")

        self.C0SpinBox.setValue(best_coeffs[0])
        self.C1SpinBox.setValue(best_coeffs[1])
        self.C2SpinBox.setValue(best_coeffs[2])
        self.C3SpinBox.setValue(best_coeffs[3])
        self.flipABCheckBox.setChecked(bool(best_flip))
        self.updateImage()

    def processImageNoPlot_torch(self, image: torch.Tensor, coeffs:torch.Tensor, scales:torch.Tensor, 
                                 AA, BB, mode: str = "bilinear") -> torch.Tensor:
        
        new_image_torch = image.clone() # type: ignore

        if self.enableCheckBox.isChecked():
            new_image_1 = new_image_torch[:,AA::2]
            new_image_1 = self.unwarp_polynomial_offset_torch(new_image_1, coeffs, scales, mode=mode)
            new_image_1 = self.unwarp_polynomial_linear_torch(new_image_1, coeffs,scales, mode=mode)
            new_image_1 = self.unwarp_polynomial_unified_torch(new_image_1,coeffs,scales, mode=mode)

            new_image_torch[:,AA::2] = new_image_1
            if self.dualEdgeCheckBox.isChecked():
                new_image_2 = new_image_torch[:,BB::2]
                coeffs = -1.0*coeffs
                new_image_2 = self.unwarp_polynomial_offset_torch(new_image_2, coeffs, scales, mode=mode)
                new_image_2 = self.unwarp_polynomial_linear_torch(new_image_2, coeffs,scales, mode=mode)
                new_image_2 = self.unwarp_polynomial_unified_torch(new_image_2, coeffs,scales, mode=mode)
                new_image_torch[:,BB::2] = new_image_2

        return new_image_torch

    def unwarp_polynomial_unified_torch(self,
        frameData :torch.Tensor ,                     # np.ndarray (H,W) or torch.Tensor (H,W)
        coeffs:torch.Tensor,                        # list/tuple/1D tensor [c0,c1,c2,c3]
        scales:torch.Tensor,                        # list/tuple (sc0, sc1, sc2, sc3)
        mode: str = "bilinear"  # interpolation mode for grid_sample ("bilinear" or "nearest")
    ) -> torch.Tensor:
        """
        PyTorch-optimized equivalent of your UnwarpPolynomialUnified using grid_sample.
        Warps columns according to the polynomial; rows (y) are identity.

        - Input:  HxW (image), either NumPy or torch
        - Output: same shape/type policy (see return_numpy_if_numpy_input)
        """
        img = frameData
        dtype, device = img.dtype, img.device

        H, W = img.shape

        y_input = torch.linspace(0.0, 1.0, H, device=device,dtype=dtype)
        # polynomial
        y_warp = ( y_input + 
            + (coeffs[2] * scales[2]) * y_input * y_input.abs()
            + (coeffs[3] * scales[3]) * (y_input ** 3) # + (coeffs[3] * scales[3]) * (y_input ** 4)
        )

        # normalize back to [0..1] and map to pixel index [0..W-1]
        denom = (y_warp.max() - y_warp.min()).clamp_min(1e-12)
        y_warp_norm = (y_warp - y_warp.min()) / denom

        # ---- build sampling grid for grid_sample ----
        # grid_sample expects normalized coords in [-1, 1]
        # x: width axis, y: height axis
        y_norm = 2.0 * y_warp_norm - 1.0          # shape (H,)
        x_norm = 2.0 * torch.arange(W, device=device, dtype=dtype) / (W - 1) - 1.0  # shape (W,)

        # make (H, W) grid: rows repeat y, columns repeat x
        grid_x = x_norm.view(1, W).expand(H, W)
        grid_y = y_norm.view(H, 1).expand(H, W)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)      # (1, H, W, 2)

        # ---- sample ----
        img_bchw = img.view(1, 1, H, W)
        out = F.grid_sample(
            img_bchw, grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=True
        )
        result = out[0, 0]  # (H, W)
        return result
    
    
    def unwarp_polynomial_offset_torch(self,
        frameData :torch.Tensor ,                     # np.ndarray (H,W) or torch.Tensor (H,W)
        coeffs:torch.Tensor,                        # list/tuple/1D tensor [c0,c1,c2,c3]
        scales:torch.Tensor,                        # list/tuple (sc0, sc1, sc2, sc3)
        mode: str = "bilinear"  # interpolation mode for grid_sample ("bilinear" or "nearest")
    ) -> torch.Tensor:
        """
        PyTorch-optimized equivalent of your UnwarpPolynomialUnified using grid_sample.
        Warps columns according to the polynomial; rows (y) are identity.

        - Input:  HxW (image), either NumPy or torch
        - Output: same shape/type policy (see return_numpy_if_numpy_input)
        """
        img = frameData
        dtype, device = img.dtype, img.device

        H, W = img.shape
        offset = 2 * scales[0] * coeffs[0] / H

        # ---- build sampling grid for grid_sample ----
        # grid_sample expects normalized coords in [-1, 1]
        # x: width axis, y: height axis
        x_norm = torch.linspace(-1.0, 1.0, W, device=device,dtype=dtype)
        y_norm = torch.linspace(-1.0, 1.0, H, device=device,dtype=dtype) + offset #over depth

        # make (H, W) grid: rows repeat y, columns repeat x
        grid_x = x_norm.view(1, W).expand(H, W)
        grid_y = y_norm.view(H, 1).expand(H, W)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)      # (1, H, W, 2)

        # ---- sample ----
        img_bchw = img.view(1, 1, H, W)
        out = F.grid_sample(
            img_bchw, grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=True
        )
        result = out[0, 0]  # (H, W)
        return result

    def unwarp_polynomial_linear_torch(self,
        frameData :torch.Tensor ,                     # np.ndarray (H,W) or torch.Tensor (H,W)
        coeffs:torch.Tensor,                        # list/tuple/1D tensor [c0,c1,c2,c3]
        scales:torch.Tensor,                        # list/tuple (sc0, sc1, sc2, sc3)
        mode: str = "bilinear"  # interpolation mode for grid_sample ("bilinear" or "nearest")
    ) -> torch.Tensor:
        """
        PyTorch-optimized equivalent of your UnwarpPolynomialUnified using grid_sample.
        Warps columns according to the polynomial; rows (y) are identity.

        - Input:  HxW (image), either NumPy or torch
        - Output: same shape/type policy (see return_numpy_if_numpy_input)
        """
        img = frameData
        dtype, device = img.dtype, img.device

        H, W = img.shape

        linear_scale = (H + (coeffs[1] * scales[1])) / H

        y_input = torch.linspace(0.0, 1.0, H, device=device,dtype=dtype)
        y_warp = linear_scale * y_input
        y_warp = 2.0 * y_warp - 1.0

        # ---- build sampling grid for grid_sample ----
        # grid_sample expects normalized coords in [-1, 1]
        # x: width axis, y: height axis
        y_norm = y_warp        # shape (H,)
        x_norm = 2.0 * torch.arange(W, device=device, dtype=dtype) / (W - 1) - 1.0  # shape (W,)

        # make (H, W) grid: rows repeat y, columns repeat x
        grid_x = x_norm.view(1, W).expand(H, W)
        grid_y = y_norm.view(H, 1).expand(H, W)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)      # (1, H, W, 2)

        # ---- sample ----
        img_bchw = img.view(1, 1, H, W)
        out = F.grid_sample(
            img_bchw, grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=True
        )
        result = out[0, 0]  # (H, W)

        return result