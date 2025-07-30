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
    
    # params for corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize = (15, 15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                                10, 0.03))

    color = np.random.randint(0, 255, (100, 3))

    img = cv.imread(image_path,cv.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv.imread(image_2_path,cv.IMREAD_GRAYSCALE) # trainImage
    
    brisk_d = cv.BRISK.create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = brisk_d.detectAndCompute(img,None)
    kp2, des2 = brisk_d.detectAndCompute(img2,None)

    p0 = cv.goodFeaturesToTrack(img, mask = None,
                             **feature_params)
    
    p2 = cv.goodFeaturesToTrack(img2, mask = None,
                             **feature_params)
    
    key_points_p0 = tuple([cv.KeyPoint(int(p.squeeze()[0]),int(p.squeeze()[1]),10) for p in p0])
    print(type(key_points_p0),len(key_points_p0),type(key_points_p0[0]))
    
    key_points_p2 = tuple([cv.KeyPoint(int(p.squeeze()[0]),int(p.squeeze()[1]),10) for p in p2])
    print(type(key_points_p2),len(key_points_p2),type(key_points_p2[0]))

    
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(img)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(img,
                                           img2,
                                           p0, None,
                                           **lk_params)
    
    

    fast_fd = cv.FastFeatureDetector.create()
    key_points = fast_fd.detect(img,None)

    print(p0.dtype,p1.dtype)
    print(type(key_points),len(key_points),type(key_points[0]))

    

    key_point_img = cv.drawKeypoints(img, key_points, None, color=(255,0,0))
    key_point_img2 = cv.drawKeypoints(img, key_points_p0, None, color=(0,255,255))
    key_point_img3 = cv.drawKeypoints(img2, key_points_p2, None, color=(255,0,255))

    cv.imshow("What?!",key_point_img)
    cv.imshow("What?! Again?!",key_point_img2)
    cv.imshow("What?! Again?! 2",key_point_img3)

    # cv.imshow("p0",p0)
    # cv.imshow("p1",p1)
    # # Select good points
    # good_new = p1[st == 1]
    # good_old = p0[st == 1]

    # # draw the tracks
    # for i, (new, old) in enumerate(zip(good_new, 
    #                                    good_old)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     mask = cv.line(mask, (a, b), (c, d),
    #                     color[i].tolist(), 2)
        
    #     img2 = cv.circle(img2, (a, b), 5,
    #                        color[i].tolist(), -1)
        
    # img = cv.add(img2, mask)

    # cv.imshow('frame', img)

    if display_in_napari:
        viewer.add_image(img)
        #viewer.add_image(mask)
        #viewer.add_image(key_point_img)
        viewer.show()
        napari.run()

feature_match.show(run=True)