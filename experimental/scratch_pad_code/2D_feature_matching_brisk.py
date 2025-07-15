""""""
from pathlib import Path

import napari
import numpy as np
import cv2 as cv
from magicgui import magicgui
import matplotlib.pyplot as plt
from napari_cool_tools_img_proc._normalization_funcs import normalize_data_in_range_func

import numpy as np
def filter_matches(kp1, kp2, matches, ratio = 0.75):
    # ratio test
    mkp1, mkp2, good= [], [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
            good.append(m)
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs, good

def explore_match(win, img1, img2, kp1, kp2, good, status = None, H = None):
    # draw homography and lines between matches
    h1, w1 = img1.shape[:2]
    matchesMask = status.ravel().tolist()
    pts = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,H)
    img2new = cv.polylines(img2, [np.int32(dst)], True,255,3, cv.LINE_AA)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img3 = cv.drawMatches(img1,kp1,img2new,kp2,good,None,**draw_params)
    return img3

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
    
    orb_d = cv.BRISK.create()
    norm = cv.NORM_HAMMING
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb_d.detectAndCompute(img,None)
    kp2, des2 = orb_d.detectAndCompute(img2,None)

    result1 = cv.drawKeypoints(img, kp1, None)
    result2 = cv.drawKeypoints(img2, kp2, None)

    # FLANN parameters
    FLANN_INDEX_LSH    = 6
    flann_params= dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 1) #2
    
    # flann = cv.FlannBasedMatcher(index_params,search_params)
    # matches = flann.knnMatch(des1,des2,k=2)

    flann_matcher = cv.FlannBasedMatcher(flann_params, {})
    raw_matches = flann_matcher.knnMatch(des1,des2,k=2)
    
    p1, p2, kp_pairs, good = filter_matches(kp1, kp2, raw_matches)
    
    if len(p1) >= 4:
        H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
    else:
        H, status = None, None
        print('%d matches found, not enough for homography estimation' % len(p1))

    vis = explore_match("win", img, img2, kp1, kp2, good, status, H)
    
    #img3 = cv.drawMatchesKnn(img,kp1,img2,kp2,matches,None,**draw_params) #FLANN
    #img3 = cv.drawMatchesKnn(img,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.imshow(vis,),plt.show()

    if display_in_napari:
        viewer.add_image(img)
        #viewer.add_image(mask)
        #viewer.add_image(key_point_img)
        viewer.show()
        napari.run()

feature_match.show(run=True)