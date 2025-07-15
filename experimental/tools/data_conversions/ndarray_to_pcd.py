""""""

""""""
from pathlib import Path

import napari
import numpy as np
from pypcd4 import PointCloud
from magicgui import magicgui


def numpy_to_pcd_format(data: np.ndarray, threshold=0):
    """"""
    if data.ndim == 2:
        data = data[:,None,:]
    x, y, z = np.where(data > threshold)
    print(f"There are {len(x)} points in the pointcloud.\n")
    # x,y,z = np.where(data != 0)
    # x,y,z = np.arange(data.shape[0],np.arange(data.shape[1]),np.arange(data.shape[2]))
    intensity_data = data[x, y, z]
    assert len(x) == len(y) == len(z) == len(intensity_data)

    return np.stack([x, y, z, intensity_data], axis=1)

@magicgui(
    image_path={"label": "Image File", "mode": "r"},
    output_dir={"label": "Output Directory", "mode": "d"},
    call_button="Feature Match",
)
def feature_match(
    image_path: Path = Path(
        r"D:\JJ\Projects\RT_Registration\Data\Test_Data\en_faces_for_comparison\associated_labels\08983951_13_09_18_deconjugated_vol_SN_prof_enface_enface [1]_labels.png"
    ),
    output_dir: Path = Path(r"D:\JJ\Projects\RT_Registration\Data\Test_Output"),
    display_in_napari: bool = False,
    #load_pcd: bool = False,
    save_pcd: bool = False,
    save_npy: bool = False,
    use_gpu: bool = True,
):
    """ """

    viewer = napari.Viewer(show=False)

    #if not load_pcd:
    viewer.open(image_path, plugin="napari")
    data = viewer.layers[-1].data
    image_name = viewer.layers[-1].name

    if save_pcd:
        pcd_numpy_data = numpy_to_pcd_format(data=data, threshold=0)
        pointcloud = PointCloud.from_xyzi_points(pcd_numpy_data)
        output_file_path = output_dir / f"{image_name}.pcd"
        pointcloud.save(output_file_path)
        print(f"{output_file_path} has been saved.")

    # if load_pcd:
    #     pcd_data:PointCloud = PointCloud.from_path(image_path)
    #     viewer.add_image(pcd_data,name="Loaded_pcd_file")

    if display_in_napari:
        # viewer.add_image(img)
        # viewer.add_image(mask)
        # viewer.add_image(key_point_img)
        viewer.show()
        napari.run()


feature_match.show(run=True)
