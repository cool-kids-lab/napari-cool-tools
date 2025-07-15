""""""
from pathlib import Path

import napari
import numpy as np
import cv2 as cv
from magicgui import magicgui
import matplotlib.pyplot as plt
from napari_cool_tools_img_proc._normalization_funcs import normalize_data_in_range_func

from stitching.images import Images
from stitching.feature_matcher import FeatureMatcher
from stitching.subsetter import Subsetter

def plot_image(img, figsize_in_inches=(5,5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()
    
def plot_images(imgs, figsize_in_inches=(5,5)):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

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
    
    imgs = [str(image_path),str(image_2_path)]
    images = Images.of(imgs)

    # Resize Images
    medium_imgs = list(images.resize(Images.Resolution.MEDIUM))
    low_imgs = list(images.resize(Images.Resolution.LOW))
    final_imgs = list(images.resize(Images.Resolution.FINAL))

    plot_images(low_imgs, (20,20))

    original_size = images.sizes[0]
    medium_size = images.get_image_size(medium_imgs[0])
    low_size = images.get_image_size(low_imgs[0])
    final_size = images.get_image_size(final_imgs[0])

    print(f"Original Size: {original_size}  -> {'{:,}'.format(np.prod(original_size))} px ~ 1 MP")
    print(f"Medium Size:   {medium_size}  -> {'{:,}'.format(np.prod(medium_size))} px ~ 0.6 MP")
    print(f"Low Size:      {low_size}   -> {'{:,}'.format(np.prod(low_size))} px ~ 0.1 MP")
    print(f"Final Size:    {final_size}  -> {'{:,}'.format(np.prod(final_size))} px ~ 1 MP")

    # Find Features
    from stitching.feature_detector import FeatureDetector

    finder = FeatureDetector()
    features = [finder.detect_features(img) for img in medium_imgs]
    keypoints_center_img = finder.draw_keypoints(medium_imgs[1], features[1])
    keypoints_center_img2 = finder.draw_keypoints(medium_imgs[0], features[0])

    plot_image(keypoints_center_img, (15,10))
    plot_image(keypoints_center_img2, (15,10))

    # Match Features
    matcher = FeatureMatcher()
    matches = matcher.match_features(features)

    matcher.get_confidence_matrix(matches)

    all_relevant_matches = matcher.draw_matches_matrix(medium_imgs, features, matches, conf_thresh=1, 
                                                   inliers=True, matchColor=(0, 255, 0))
    
    # Subset
    subsetter = Subsetter()
    dot_notation = subsetter.get_matches_graph(images.names, matches)
    print(dot_notation)

    for idx1, idx2, img in all_relevant_matches:
        print(f"Matches Image {idx1+1} to Image {idx2+1}")
        plot_image(img, (20,10))

    indices = subsetter.get_indices_to_keep(features, matches)

    medium_imgs = subsetter.subset_list(medium_imgs, indices)
    low_imgs = subsetter.subset_list(low_imgs, indices)
    final_imgs = subsetter.subset_list(final_imgs, indices)
    features = subsetter.subset_list(features, indices)
    matches = subsetter.subset_matches(matches, indices)

    images.subset(indices)

    print(images.names)
    print(matcher.get_confidence_matrix(matches))

    # if display_in_napari:
    #     viewer.add_image(img)
    #     #viewer.add_image(mask)
    #     #viewer.add_image(key_point_img)
    #     viewer.show()
    #     napari.run()

feature_match.show(run=True)