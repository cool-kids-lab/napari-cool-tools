""""""
from pathlib import Path

import napari
import numpy as np
from scipy.ndimage import grey_opening
from skimage.filters import meijering,sato,frangi,hessian
from skimage.morphology import disk
import torch
from torchvision import io
from torchvision.transforms.v2.functional import resize, InterpolationMode, gaussian_blur
from kornia.morphology import dilation,erosion,opening,closing
import cv2 as cv
from magicgui import magicgui
import matplotlib.pyplot as plt
from napari_cool_tools_img_proc._normalization_funcs import normalize_data_in_range_func
import imreg_dft as ird


@magicgui(
    image_path={"label": "Image File", "mode": "r"},
    image_2_path={"label": "Image File 2", "mode": "r"},
    output_dir={"label": "Output Directory", "mode": "d"},
    call_button="Feature Match",
)
def feature_match(
    image_path: Path = Path(
        r"D:\JJ\Projects\RT_Registration\Data\Test_Data\en_faces_for_comparison\processed_means_with_labels\masked_means\13_09_26_masked.png"
    ),
    # image_mask_path: Path = Path(
    #     r"F:\Registration Sample\Test_Data\All_Peripheral_for registration\2024.09.18\08983951\en_faces_for_comparison\08983951_13_09_18_deconjugated_vol_SN_prof_enface_enface_labels.png"
    # ),
    image_2_path: Path = Path(
        r"D:\JJ\Projects\RT_Registration\Data\Test_Data\en_faces_for_comparison\processed_means_with_labels\masked_means\13_09_32_masked.png"
    ),
    # image_2_mask_path: Path = Path(
    #     r"F:\Registration Sample\Test_Data\All_Peripheral_for registration\2024.09.18\08983951\en_faces_for_comparison\08983951_13_09_32_deconjucated_vol_SN_prof_enface_enface_labels.png"
    # ),
    output_dir: Path = Path(r"D:\JJ\Projects\RT_Registration\Data\Test_Output"),
    display_in_napari: bool = True,
    upscale_factor:int = 0,
    disc_radius:float = 1,
    opening_iter:int = 10,
    gaussian:bool = True,
    use_opening:bool = True,
):
    """ """
    if display_in_napari:
        viewer = napari.Viewer(show=False)
    
    img = io.read_image(image_path) #.squeeze().numpy()
    img2 = io.read_image(image_2_path) #.squeeze().numpy()

    out = img
    out2 = img2

    # step 1 preprocess retinal image
    # Upscale image 
    if upscale_factor:
        new_shape = [upscale_factor*shap for shap in img.shape[-2:]]
        new_shape2 = [upscale_factor*shap for shap in img2.shape[-2:]]
        out = resize(torch.tensor(img),size=new_shape,interpolation=InterpolationMode.BICUBIC) #.squeeze().numpy()
        out2 = resize(torch.tensor(img2),size=new_shape2,interpolation=InterpolationMode.BICUBIC) #.squeeze().numpy()

    #disc_kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
    #disc_kernel = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]])
    disc_kernel = disk(radius=disc_radius)

    if use_opening:
        #opening_iter = 10
        for idx in range(opening_iter):
            out = torch.tensor(grey_opening(out.squeeze().numpy(),structure=disc_kernel),requires_grad=False).unsqueeze(0)
            out2 = torch.tensor(grey_opening(out2.squeeze().numpy(),structure=disc_kernel),requires_grad=False).unsqueeze(0)

    if gaussian:
        out = gaussian_blur(out,3)
        out2 = gaussian_blur(out2,3)

    #meijering,sato,frangi,hessian

    #out = hessian(out.squeeze().numpy(),[1])
    #out2 = hessian(out2.squeeze().numpy(),[1])

    #out = hessian(out.squeeze().numpy(),sigmas=range(1,8))
    #out2 = hessian(out2.squeeze().numpy(),sigmas=range(1,8))

    # step 2 Compute ridge image


    if display_in_napari:
        viewer.add_image(img,name="template")
        viewer.add_image(img2,name="to_transform")
        viewer.add_image(out,name="template_out")
        viewer.add_image(out2,name="to_transform_out")
        viewer.show()
        napari.run()

feature_match.show(run=True)