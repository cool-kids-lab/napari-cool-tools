""" """

from pathlib import Path

import napari
from magicgui import magicgui

# import pptk
# import open3d as o3d
# from pypcd4 import PointCloud
# from napari_cool_tools_io

# from napari_cool_tools_img_proc._equalization_funcs import init_bscan_preproc, DTYPE


@magicgui(
    fold_dir={"label": "Fold Directory", "mode": "d"},
    output_dir={"label": "Output Directory", "mode": "d"},
    call_button="Generate Training Folds",
)
def generate_point_cloud(
    fold_dir: Path = Path(r"D:\JJ\Projects\Segmentation_Paper\Data\Bscan"),
    output_dir: Path = Path(r"D:\JJ\Projects\Segmentation_Paper\Data\Bscan"),
):
    """"""

    file_paths = list(fold_dir.rglob("*_Images.pt"))
    test_file_path = file_paths[0]

    viewer = napari.Viewer(show=False)
    viewer.open(test_file_path, plugin="napari-cool-tools-io")

    # test_numpy_data = viewer.layers[-1].data
    #
    # x,y,z = np.where(test_numpy_data != 0)
    # intensities = test_numpy_data[x,y,z]
    #
    # assert len(x) == len(y) == len(z) == len(intensities)
    #
    # pcd_numpy_data = np.stack([x,y,z,intensities],axis=1)
    #
    # print(f"\npcd_numpy_data shape: {pcd_numpy_data.shape}\n\n")
    #
    # test_pcd = PointCloud.from_xyzi_points(pcd_numpy_data)
    #
    ##test_pcd = o3d.t.geometry.PointCloud(test_numpy_data)
    #
    # print(test_pcd,"\n",type(test_pcd),"\n\n")

    # test_pcd.save("./test_data.pcd")

    # o3d_pcd = o3d.io.read_point_cloud("./test_data.pcd")
    #
    # print(f"O3d data:\n")
    # print(o3d_pcd,"\n",type(o3d_pcd),"\n\n")

    # x = np.random(100,3)

    # v = pptk.viewer(x)
    # v.set(point_size=0.01)

    viewer.show()
    napari.run()

    # o3d.visualization.draw_geometries([o3d_pcd])


# view_bscan_variants.changed.connect(print)
generate_point_cloud.show(run=True)
