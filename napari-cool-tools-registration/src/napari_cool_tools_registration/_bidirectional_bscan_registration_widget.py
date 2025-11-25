# This Python file uses the following encoding: utf-8

from qtpy.QtWidgets import QDialog
from qtpy import QtWidgets
from napari_cool_tools_registration._bidirectional_bscan_registration_form import Ui_Dialog
import pyqtgraph as pg
import numpy as np
import torch
import torch.nn.functional as F
from napari_cool_tools_oct_preproc._oct_preproc_func import desine
from napari_cool_tools_io import viewer, device
from tqdm import tqdm

def blur_score_vol_torch_frequency(img: torch.Tensor):
    """
    Edge magnitude via frequency-domain derivatives.
    img: (H, W) real tensor (any float dtype); returns a scalar score (sum of magnitudes).
    """
    assert img.ndim == 2, "Input must be 2D (H, W)."
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

def blur_score_vol_torch_spatial(I: torch.Tensor) -> torch.Tensor:
    """
    Sum of absolute Laplacian (higher => sharper).
    Returns a *torch scalar* (zero-dim tensor) on the same device, float64.

    I: (H,W) or (D,H,W). For 3D, applies 2D Laplacian per slice.
    """
    assert I.ndim in (2, 3), "I must be (H,W) or (D,H,W)"
    x = I.to(torch.float64)

    # Shape to (N,1,H,W) for conv2d
    x4 = x.unsqueeze(0).unsqueeze(0) if x.ndim == 2 else x.unsqueeze(1)

    # 3x3 Laplacian kernel
    h = torch.tensor([[0., 1., 0.],
                      [1., -4., 1.],
                      [0., 1., 0.]], dtype=torch.float64, device=x.device).view(1,1,3,3)

    # replicate-pad edges (nearest) then conv
    xpad = F.pad(x4, (1, 1, 1, 1), mode='replicate')
    L = F.conv2d(xpad, h)  # (N,1,H,W)

    # Sum of absolute Laplacian -> torch scalar
    score = L.abs().sum()
    return score


class Bidirectional_Bscan_Registration_Widget(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Bidirectional Bscan Registration Dialog")

        #initialize variables
        self.volume = None
        self.enface = None

        # Create pyqtgraph image viewer
        # pg.setConfigOption('imageAxisOrder', 'row-major')
        self.axes = {'x':1, 'y':0} #this is inverse because napari transposes images
        self.viewer = pg.ImageView(parent=self)
        self.viewer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.viewer.updateGeometry()

        layout = self.graphicsViewPlaceHolder.parent().layout()
        layout.replaceWidget(self.graphicsViewPlaceHolder, self.viewer)
        self.graphicsViewPlaceHolder.deleteLater()

        self.viewer.ui.roiBtn.hide()
        self.viewer.ui.menuBtn.hide()
        self.viewer.ui.histogram.hide()

        #show a random image
        self.first_time = True
        bscan = np.zeros((256, 256))
        self.viewer.setImage(bscan, autoRange=True, autoLevels = False, levels=[self.minSpinBox.value(),self.maxSpinBox.value()]
                             , axes=self.axes)

        self.maxSpinBox.valueChanged.connect(self.updateImage)
        self.minSpinBox.valueChanged.connect(self.updateImage)
        self.predesineCheckBox.stateChanged.connect(self.updateImage)
        self.postdesineCheckBox.stateChanged.connect(self.updateImage)
        self.enableCheckBox.stateChanged.connect(self.updateImage)
        self.flipABCheckBox.stateChanged.connect(self.updateImage)
        self.dualEdgeCheckBox.stateChanged.connect(self.updateImage)
        self.linearInterpCheckBox.stateChanged.connect(self.updateImage)
        self.cropCheckBox.stateChanged.connect(self.updateImage)
        self.inverseCheckBox.stateChanged.connect(self.updateImage)

        self.C0ScaleComboBox.currentTextChanged.connect(self.updateImage)
        self.C1ScaleComboBox.currentTextChanged.connect(self.updateImage)
        self.C2ScaleComboBox.currentTextChanged.connect(self.updateImage)
        self.C3ScaleComboBox.currentTextChanged.connect(self.updateImage)
        self.C0SpinBox.valueChanged.connect(self.updateImage)
        self.C1SpinBox.valueChanged.connect(self.updateImage)
        self.C2SpinBox.valueChanged.connect(self.updateImage)
        self.C3SpinBox.valueChanged.connect(self.updateImage)
        self.autoFindPushButton.clicked.connect(self.autoFindCoeffs)

        self.clipSpinBox.setValue(10)
        self.mipCheckBox.stateChanged.connect(self.updateEnface)
        self.clipSpinBox.valueChanged.connect(self.updateEnface)

        self.viewer.ui.roiBtn.hide()
        self.viewer.ui.menuBtn.hide()
        self.viewer.ui.histogram.hide()

        self.updateImage()

    def updateImage(self):

        if self.enface is None:
            return

        new_image_torch = torch.from_numpy(self.enface.copy()).to(device=device)

        if self.predesineCheckBox.isChecked():
            new_image_torch = desine(new_image_torch, transpose=False, scale_fac=1)

        if self.linearInterpCheckBox.isChecked():
            mode = "bilinear"
        else:
            mode = "nearest"

        if self.flipABCheckBox.isChecked():
            AA, BB = (1, 0)
        else:
            AA, BB = (0, 1)

        if self.enableCheckBox.isChecked():

            coeffs = [self.C0SpinBox.value(), self.C1SpinBox.value(), self.C2SpinBox.value(), self.C3SpinBox.value()]
            scales = [float(self.C0ScaleComboBox.currentText()), float(self.C1ScaleComboBox.currentText()),
                    float(self.C2ScaleComboBox.currentText()), float(self.C3ScaleComboBox.currentText())]
            
            coeffs = torch.as_tensor(coeffs, dtype=new_image_torch.dtype, device=device)
            scales = torch.as_tensor(scales, dtype=new_image_torch.dtype, device=device)

            new_image_1 = new_image_torch[AA::2,:]
            new_image_1 = self.unwarp_polynomial_offset_torch(new_image_1, coeffs, scales, mode=mode)
            new_image_1 = self.unwarp_polynomial_linear_torch(new_image_1, coeffs, scales, mode=mode,
                                                              inverse=self.inverseCheckBox.isChecked(),
                                                              centered=self.centeredCheckBox.isChecked())
            new_image_1 = self.unwarp_polynomial_unified_torch(new_image_1, coeffs, scales, mode=mode)
            new_image_torch[AA::2,:] = new_image_1

            if self.dualEdgeCheckBox.isChecked():
                new_image_2 = new_image_torch[BB::2,:]
                coeffs = coeffs*-1.0
                new_image_2 = self.unwarp_polynomial_offset_torch(new_image_2, coeffs, scales, mode=mode)
                new_image_2 = self.unwarp_polynomial_linear_torch(new_image_2, coeffs, scales, mode=mode,
                                                                  inverse=self.inverseCheckBox.isChecked(),
                                                                  centered=self.centeredCheckBox.isChecked())
                new_image_2 = self.unwarp_polynomial_unified_torch(new_image_2, coeffs, scales, mode=mode)
                new_image_torch[BB::2,:] = new_image_2

            #crop the sides to avoid edge artifacts
            if self.cropCheckBox.isChecked():
                crop_size = int(abs(coeffs[0]*scales[0]))
                new_image_torch[:, :crop_size] = 0.0       # left side
                H, W = new_image_torch.shape
                new_image_torch[:, W-crop_size:] = 0.0      # right side

        if self.postdesineCheckBox.isChecked():
            new_image_torch = desine(new_image_torch, transpose=False, scale_fac=1)

        new_image = new_image_torch.cpu().numpy()

        if self.first_time:
            self.viewer.setImage(new_image, autoRange=True, autoLevels = False, levels=[self.minSpinBox.value(),self.maxSpinBox.value()],
                                axes=self.axes)
            self.first_time = False
        else:
            self.viewer.setImage(new_image, autoRange=False, autoLevels = False, levels=[self.minSpinBox.value(),self.maxSpinBox.value()], axes=self.axes)
        
    
    def updateEnface(self):
        if self.volume is None:
            return
        
        clip_value = self.clipSpinBox.value()

        # Apply clipping safely (handles 0 correctly)
        if clip_value > 0:
            clipped = self.volume[:, clip_value:-clip_value, :] #crop the top and bottom
        else:
            clipped = self.volume

        if self.mipCheckBox.isChecked():
            self.enface = np.max(clipped, axis=1)
        else:
            self.enface = np.mean(clipped, axis=1)

        vmin, vmax = np.percentile(self.enface, (1, 99))
        self.minSpinBox.setValue(float(vmin))
        self.maxSpinBox.setValue(float(vmax))
    
    def get_output_volume(self):

        print("Getting output volume...")
        #TODO: implement the full 3D volume processing here
        if self.volume is None:
            return None

        output_volume = np.zeros_like(self.volume)

        # print(self.volume.shape)

        volume_torch = torch.from_numpy(self.volume.copy()).to(device=device)

        # print(volume_torch.shape)

        if self.predesineCheckBox.isChecked():
            volume_torch = desine(volume_torch, transpose=False, scale_fac=2)

        if self.linearInterpCheckBox.isChecked():
            mode = "bilinear"
        else:
            mode = "nearest"

        if self.flipABCheckBox.isChecked():
            AA, BB = (1, 0)
        else:
            AA, BB = (0, 1)

        coeffs = [self.C0SpinBox.value(), self.C1SpinBox.value(), self.C2SpinBox.value(), self.C3SpinBox.value()]
        scales = [float(self.C0ScaleComboBox.currentText()), float(self.C1ScaleComboBox.currentText()),
        float(self.C2ScaleComboBox.currentText()), float(self.C3ScaleComboBox.currentText())]

        coeffs = torch.as_tensor(coeffs, dtype=volume_torch.dtype, device=device)
        scales = torch.as_tensor(scales, dtype=volume_torch.dtype, device=device)

        for i in tqdm(range(self.volume.shape[1]), desc="Processing B-scans"):
            bscan_torch = volume_torch[:,i,:]

            # print(bscan_torch.shape)

            if self.enableCheckBox.isChecked():
                new_image_1 = bscan_torch[AA::2,:]
                new_image_1 = self.unwarp_polynomial_offset_torch(new_image_1, coeffs, scales, mode=mode)
                new_image_1 = self.unwarp_polynomial_linear_torch(new_image_1, coeffs, scales, mode=mode,
                                                                inverse=self.inverseCheckBox.isChecked(),
                                                                centered=self.centeredCheckBox.isChecked())
                new_image_1 = self.unwarp_polynomial_unified_torch(new_image_1, coeffs, scales, mode=mode)
                bscan_torch[AA::2,:] = new_image_1

                if self.dualEdgeCheckBox.isChecked():
                    new_image_2 = bscan_torch[BB::2,:]
                    coeffs_inv = coeffs*-1.0
                    new_image_2 = self.unwarp_polynomial_offset_torch(new_image_2, coeffs_inv, scales, mode=mode)
                    new_image_2 = self.unwarp_polynomial_linear_torch(new_image_2, coeffs_inv, scales, mode=mode,
                                                                    inverse=self.inverseCheckBox.isChecked(),
                                                                    centered=self.centeredCheckBox.isChecked())
                    new_image_2 = self.unwarp_polynomial_unified_torch(new_image_2, coeffs_inv, scales, mode=mode)
                    bscan_torch[BB::2,:] = new_image_2

                # #crop the sides to avoid edge artifacts
                # if self.cropCheckBox.isChecked():
                #     crop_size = int(abs(coeffs[0]*scales[0]))
                #     bscan_torch[:, :crop_size] = 0.0       # left side
                #     H, W = bscan_torch.shape
                #     bscan_torch[:, W-crop_size:] = 0.0      # right side
                                                                        
            volume_torch[:,i,:] = bscan_torch
        
        if self.postdesineCheckBox.isChecked():
                volume_torch = desine(volume_torch, transpose=False, scale_fac=2)

        output_volume = volume_torch.cpu().numpy()

        return output_volume

    def set_input_volume(self, volume: np.ndarray):
        self.volume = volume
        self.updateEnface()

    def autoFindCoeffs(self):

        if self.enface is None:
            return

        image = torch.from_numpy(self.enface.copy()).to(device=device)

        if self.predesineCheckBox.isChecked():
            image = desine(image, transpose=False, scale_fac=1)

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
            c0_range = np.arange(-ranges, ranges, 1, dtype=np.float64)*step_size + c0_current

        if self.C1CheckBox.isChecked():
            c1_range = np.arange(-ranges, ranges, 1, dtype=np.float64)*step_size + c1_current

        if self.C2CheckBox.isChecked():
            c2_range = np.arange(-ranges, ranges, 1, dtype=np.float64)*step_size + c2_current

        if self.C3CheckBox.isChecked():
            c3_range = np.arange(-ranges, ranges, 1, dtype=np.float64)*step_size + c3_current

        total_iterations = (
            len(c0_range) * len(c1_range) * len(c2_range) * len(c3_range)
        )

        best_score = 0
        best_coeffs = torch.as_tensor([c0_current, c1_current, c2_current, c3_current], dtype=torch.float64, device="cuda")

        sc0 = float(self.C0ScaleComboBox.currentText())
        sc1 = float(self.C1ScaleComboBox.currentText())
        sc2 = float(self.C2ScaleComboBox.currentText())
        sc3 = float(self.C3ScaleComboBox.currentText())

        scales = torch.as_tensor([sc0, sc1, sc2, sc3], dtype=torch.float64, device="cuda")

        if self.linearInterpCheckBox.isChecked():
            mode = "bilinear"
        else:
            mode = "nearest"

        AA, BB = (0, 1)

        # initialize score
        new_image_torch = self.processImageNoPlot_torch(image,best_coeffs,scales,AA=AA,BB=BB,mode=mode)

        if self.frequencyDomainCheckBox.isChecked():
            blur_score_vol = blur_score_vol_torch_frequency
        else:
            blur_score_vol = blur_score_vol_torch_spatial

        best_score = blur_score_vol(new_image_torch).item()

        inverse = self.inverseCheckBox.isChecked()
        centered = self.centeredCheckBox.isChecked()

        iteration = 0
        with tqdm(total=total_iterations, desc="Searching coeffs") as pbar:
            pbar.set_postfix(best_score=best_score,coeffs=best_coeffs)
            for c0 in c0_range:
                for c1 in c1_range:
                    for c2 in c2_range:
                        for c3 in c3_range:
                            iteration += 1
                            coeffs = torch.as_tensor([c0, c1, c2, c3], dtype=image.dtype, device=device)
                            new_image_torch = self.processImageNoPlot_torch(image,coeffs,scales,AA=AA,BB=BB,mode=mode,
                                                                            inverse=inverse,
                                                                          centered=centered)
                            score = blur_score_vol(new_image_torch).item()
                            if score < best_score:
                                best_score = score
                                best_coeffs = [c0, c1, c2, c3]
                                pbar.set_postfix(
                                    best_score=best_score,
                                    coeffs=best_coeffs
                                )

                            pbar.update(1)

        print(f"Best coeffs found: {best_coeffs} with score: {best_score}")

        self.C0SpinBox.setValue(best_coeffs[0])
        self.C1SpinBox.setValue(best_coeffs[1])
        self.C2SpinBox.setValue(best_coeffs[2])
        self.C3SpinBox.setValue(best_coeffs[3])
        self.updateImage()

    def processImageNoPlot_torch(self, image: torch.Tensor, coeffs:torch.Tensor, scales:torch.Tensor, 
                                 AA, BB, mode: str = "bilinear",
                                 inverse: bool = False,
                                 centered: bool = False) -> torch.Tensor:

        new_image_torch = image.clone() # type: ignore

        if self.enableCheckBox.isChecked():            
            new_image_1 = new_image_torch[AA::2,:]
            new_image_1 = self.unwarp_polynomial_offset_torch(new_image_1, coeffs, scales, mode=mode)
            new_image_1 = self.unwarp_polynomial_linear_torch(new_image_1, coeffs,scales, mode=mode,
                                                              inverse=inverse,
                                                                centered=centered)
            new_image_1 = self.unwarp_polynomial_unified_torch(new_image_1,coeffs,scales, mode=mode)
            new_image_torch[AA::2,:] = new_image_1
            
            if self.dualEdgeCheckBox.isChecked():
                new_image_2 = new_image_torch[BB::2,:]
                coeffs = coeffs*-1.0
                new_image_2 = self.unwarp_polynomial_offset_torch(new_image_2, coeffs, scales, mode=mode)
                new_image_2 = self.unwarp_polynomial_linear_torch(new_image_2, coeffs,scales, mode=mode,
                                                                  inverse=inverse,
                                                                    centered=centered)
                new_image_2 = self.unwarp_polynomial_unified_torch(new_image_2, coeffs,scales, mode=mode)
                new_image_torch[BB::2,:] = new_image_2

            #crop the sides to avoid edge artifacts
            if self.cropCheckBox.isChecked():
                crop_size = int(abs(coeffs[0]*scales[0]))
                new_image_torch[:, :crop_size] = 0.0       # left side
                H, W = new_image_torch.shape
                new_image_torch[:, W-crop_size:] = 0.0      # right side

        return new_image_torch
    
    def unwarp_polynomial_unified_torch(self,
        img :torch.Tensor ,                     # np.ndarray (H,W) or torch.Tensor (H,W)
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

        H, W = img.shape

        x_input = torch.linspace(0.0, 1.0, W, device=img.device,dtype=img.dtype)

        # polynomial
        x_in = x_input
        x_warp = ( x_in + 
            + (coeffs[2] * scales[2]) * x_in * x_in.abs()
            + (coeffs[3] * scales[3]) * (x_in ** 3) # + (coeffs[3] * scales[3]) * (x_in ** 4)
        )

        # normalize back to [0..1] and map to pixel index [0..W-1]
        denom = (x_warp.max() - x_warp.min()).clamp_min(1e-12)
        x_warp_norm = (x_warp - x_warp.min()) / denom

        # ---- build sampling grid for grid_sample ----
        # grid_sample expects normalized coords in [-1, 1]
        # x: width axis, y: height axis
        x_norm = 2.0 * x_warp_norm - 1.0          # shape (W,)
        y_norm = 2.0 * torch.arange(H, device=img.device, dtype=img.dtype) / (H - 1) - 1.0  # shape (H,)

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
        img :torch.Tensor ,                     # np.ndarray (H,W) or torch.Tensor (H,W)
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

        H, W = img.shape
        offset = 2 * scales[0] * coeffs[0] / W

        # ---- build sampling grid for grid_sample ----
        # grid_sample expects normalized coords in [-1, 1]
        # x: width axis, y: height axis
        x_norm = torch.linspace(-1.0, 1.0, W, device=img.device,dtype=img.dtype) + offset
        y_norm = torch.linspace(-1.0, 1.0, H, device=img.device,dtype=img.dtype)

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
        img :torch.Tensor ,                     # np.ndarray (H,W) or torch.Tensor (H,W)
        coeffs:torch.Tensor,                        # list/tuple/1D tensor [c0,c1,c2,c3]
        scales:torch.Tensor,                        # list/tuple (sc0, sc1, sc2, sc3)
        mode: str = "bilinear",  # interpolation mode for grid_sample ("bilinear" or "nearest"),
        inverse: bool = False,
        centered: bool = False,
    ) -> torch.Tensor:
        """
        PyTorch-optimized equivalent of your UnwarpPolynomialUnified using grid_sample.
        Warps columns according to the polynomial; rows (y) are identity.

        - Input:  HxW (image), either NumPy or torch
        - Output: same shape/type policy (see return_numpy_if_numpy_input)
        """
        if inverse:
            img = torch.flip(img, dims=[1])

        H, W = img.shape

        linear_scale = (W + (coeffs[1] * scales[1])) / W

        if centered:
            x_input = torch.linspace(-1.0, 1.0, W, device=img.device,dtype=img.dtype)
            x_warp = linear_scale * x_input
        else:
            x_input = torch.linspace(0.0, 1.0, W, device=img.device,dtype=img.dtype)
            x_warp = linear_scale * x_input
            x_warp = 2.0 * x_warp - 1.0

        # ---- build sampling grid for grid_sample ----
        # grid_sample expects normalized coords in [-1, 1]
        # x: width axis, y: height axis
        x_norm = x_warp        # shape (W,)
        y_norm = 2.0 * torch.arange(H, device=img.device, dtype=img.dtype) / (H - 1) - 1.0  # shape (H,)

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

        if inverse:
            result = torch.flip(result, dims=[1])

        return result