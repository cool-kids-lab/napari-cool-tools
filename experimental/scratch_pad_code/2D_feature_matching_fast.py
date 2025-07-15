""""""
from pathlib import Path

import napari
import numpy as np
import cv2 as cv
from magicgui import magicgui
import matplotlib.pyplot as plt
from napari_cool_tools_img_proc._normalization_funcs import normalize_data_in_range_func


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
    output_filename: str = "output.pt",
    display_in_napari: bool = False,
    save_pcd: bool = False,
    save_npy: bool = False,
    use_gpu: bool = True,

):
    """ """
    if display_in_napari:
        viewer = napari.Viewer(show=False)
    
    img = cv.imread(image_path,cv.IMREAD_GRAYSCALE)
    #mask = cv.imread(image_mask_path,cv.IMREAD_GRAYSCALE)
    #mask = normalize_data_in_range_func(mask).astype(np.float64)
    #img = mask

    fast_fd = cv.FastFeatureDetector.create()
    key_points = fast_fd.detect(img,None)
    key_point_img = cv.drawKeypoints(img, key_points, None, color=(255,0,0))

    # Print all default params
    print( "Threshold: {}".format(fast_fd.getThreshold()) )
    print( "nonmaxSuppression:{}".format(fast_fd.getNonmaxSuppression()) )
    print( "neighborhood: {}".format(fast_fd.getType()) )
    print( "Total Keypoints with nonmaxSuppression: {}".format(len(key_points)) )

    if display_in_napari:
        viewer.add_image(img)
        #viewer.add_image(mask)
        viewer.add_image(key_point_img)
        viewer.show()
        napari.run()

feature_match.show(run=True)