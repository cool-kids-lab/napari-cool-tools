""""""
from pathlib import Path

import napari
import numpy as np
import pyvista as pv
import cv2 as cv
from magicgui import magicgui
import matplotlib.pyplot as plt
from napari_cool_tools_img_proc._normalization_funcs import normalize_data_in_range_func
import imreg_dft as ird

@magicgui(
    image_path={"label": "Image File", "mode": "r"},
    image2_path={"label": "Image File 2", "mode": "r"},
    #output_dir={"label": "Output Directory", "mode": "d"},
    call_button="Feature Match",
)
def feature_match(
    image_path: Path = Path(
        r"D:\JJ\Projects\RT_Registration\Data\Test_Data\en_faces_for_comparison\processed_means_with_labels\masked_means\13_09_26_masked.png"
    ),
    # image_mask_path: Path = Path(
    #     r"F:\Registration Sample\Test_Data\All_Peripheral_for registration\2024.09.18\08983951\en_faces_for_comparison\08983951_13_09_18_deconjugated_vol_SN_prof_enface_enface_labels.png"
    # ),
    image2_path: Path = Path(
        r"D:\JJ\Projects\RT_Registration\Data\Test_Data\en_faces_for_comparison\processed_means_with_labels\masked_means\13_09_32_masked.png"
    ),
    # image_2_mask_path: Path = Path(
    #     r"F:\Registration Sample\Test_Data\All_Peripheral_for registration\2024.09.18\08983951\en_faces_for_comparison\08983951_13_09_32_deconjucated_vol_SN_prof_enface_enface_labels.png"
    # ),
    #output_dir: Path = Path(r"D:\JJ\Projects\RT_Registration\Data\Test_Output"),
    #output_filename: str = "output.pt",
    display_in_napari: bool = False,
    #save_pcd: bool = False,
    #save_npy: bool = False,
    #use_gpu: bool = True,
    num_iter:int=3,
):
    """ """
    if display_in_napari:
        viewer = napari.Viewer(show=False)
    
    # img = cv.imread(image_path,cv.IMREAD_GRAYSCALE)
    # #mask = cv.imread(image_mask_path,cv.IMREAD_GRAYSCALE)
    # #mask = normalize_data_in_range_func(mask).astype(np.float64)
    # #img = mask

    # #fast_fd = cv.FastFeatureDetector.create()
    # #key_points = fast_fd.detect(img,None)
    # #key_point_img = cv.drawKeypoints(img, key_points, None, color=(255,0,0))

    img = cv.imread(image_path,cv.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv.imread(image2_path,cv.IMREAD_GRAYSCALE) # trainImage

    result = ird.similarity(img,img2,numiter=num_iter)
    #print(f"Reuslts are:\n{result}\n")
    print(f"\n\n{result.keys()}\n\n")
    print(f"Reuslts are:\nsuccess prct: {result['success']}\ntranslation vector: {result['tvec']}\nangle: {result['angle']}\nscale: {result['scale']}\n")
    print(f"Reuslts are:\nDt: {result['Dt']}\nDangle: {result['Dangle']}\nDscale: {result['Dscale']}\n")
    

    assert "timg" in result
    timg = result['timg']

    #pv_img = pv.read(image_path)
    #pv_img2 = pv.read(image2_path)

    enface = pv.Plane(i_size=8.4,j_size=8.0)
    enface2 = pv.Plane(i_size=8.4,j_size=8.0)
    enface3 = pv.Plane(i_size=8.4,j_size=8.0)

    enface.texture_map_to_plane(inplace=True)
    enface2.texture_map_to_plane(inplace=True)
    enface3.texture_map_to_plane(inplace=True)

    plotter = pv.Plotter()
    actor = plotter.add_mesh(enface,texture=img)
    actor2 = plotter.add_mesh(enface2,texture=img2)
    actor3 = plotter.add_mesh(enface3,texture=timg)
    #plotter.add_actor(pv_img)
    #plotter.add_actor(pv_img2)
    widget = plotter.add_affine_transform_widget(actor)
    widget2 = plotter.add_affine_transform_widget(actor2)
    widget3 = plotter.add_affine_transform_widget(actor3)
    plotter.show()

    if display_in_napari:
        viewer.add_image(img,name="template")
        viewer.add_image(img2,name="to_transform")
        viewer.add_image(result["timg"],name="transformed")
        #viewer.add_image(mask)
        #viewer.add_image(key_point_img)
        viewer.show()
        napari.run()

feature_match.show(run=True)