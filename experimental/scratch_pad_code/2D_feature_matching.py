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
    
    # img = cv.imread(image_path,cv.IMREAD_GRAYSCALE)
    # #mask = cv.imread(image_mask_path,cv.IMREAD_GRAYSCALE)
    # #mask = normalize_data_in_range_func(mask).astype(np.float64)
    # #img = mask

    # #fast_fd = cv.FastFeatureDetector.create()
    # #key_points = fast_fd.detect(img,None)
    # #key_point_img = cv.drawKeypoints(img, key_points, None, color=(255,0,0))

    img = cv.imread(image_path,cv.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv.imread(image_2_path,cv.IMREAD_GRAYSCALE) # trainImage
    
    # Initiate SIFT detector
    sift = cv.SIFT.create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    
    # flann = cv.FlannBasedMatcher(index_params,search_params)
    # matches = flann.knnMatch(des1,des2,k=2)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    
    # # FLANN
    # # ratio test as per Lowe's paper
    # for i,(m,n) in enumerate(matches):
    #     if m.distance < 0.7*n.distance:
    #         matchesMask[i]=[1,0]

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)
    
    #img3 = cv.drawMatchesKnn(img,kp1,img2,kp2,matches,None,**draw_params) #FLANN
    img3 = cv.drawMatchesKnn(img,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.imshow(img3,),plt.show()

    if display_in_napari:
        viewer.add_image(img)
        #viewer.add_image(mask)
        #viewer.add_image(key_point_img)
        viewer.show()
        napari.run()

feature_match.show(run=True)