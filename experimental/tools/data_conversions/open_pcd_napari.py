""""""

""""""
from pathlib import Path

import napari
import numpy as np
import pyvista as pv
from pypcd4 import PointCloud
from magicgui import magicgui

img_path = Path(r"D:\JJ\Projects\RT_Registration\Data\Test_Output\08983951_13_09_18_deconjugated_vol_SN_prof_enface_enface [1]_labels.pcd")

def pcd_to_ndarray(pcd_data:PointCloud):
    """"""
    # List comprehension to separate xyz and intensity 
    xyzi = [pcd_numpy_data[:,idx].astype(int) for idx in range(len(pcd_numpy_data[0,:]))]
    #np.meshgrid(*xyzi[:-1],indexing='xy',sparse=True)
    xyz = xyzi[:-1]
    i = xyzi[-1]
    np_data = np.zeros((840,1,800))
    np_data[xyz] = 1
    return np_data.squeeze()

pcd_data:PointCloud = PointCloud.from_path(img_path)
pcd_numpy_data = pcd_data.numpy()

@magicgui(
    image_path={"label": "Image File", "mode": "r"},
    image_path2={"label": "Image File", "mode": "r"},
    call_button="Feature Match",
)
def feature_match(
    image_path: Path = Path(
        r"D:\JJ\Projects\RT_Registration\Data\Test_Output\08983951_13_09_18_deconjugated_vol_SN_prof_enface_enface [1]_labels.pcd"
    ),
    image_path2: Path = Path(
        r"D:\JJ\Projects\RT_Registration\Data\Test_Output\08983951_13_09_20_deconjucated_vol_SN_prof_enface_enface [1]_labels.pcd"
    ),
):
    """ """
    # viewer = napari.Viewer(show=False)

    # pcd_data:PointCloud = PointCloud.from_path(image_path)
    # np_data = pcd_to_ndarray(pcd_data)
    # viewer.add_image(np_data,name="Loaded_pcd_file")

    # viewer.show()
    # napari.run()

    src_pc : PointCloud = PointCloud.from_path(image_path)
    tgt_pc : PointCloud = PointCloud.from_path(image_path2)
    print(src_pc.fields,tgt_pc.fields)

    src_np = src_pc.numpy()
    tgt_np = tgt_pc.numpy()
    print(f"src vs tgt shapes: {src_np.shape,tgt_np.shape}\n")

    src_point_cloud = pv.PolyData(src_np[:,:3])
    tgt_point_cloud = pv.PolyData(tgt_np[:,:3])

    src_point_cloud["intensity"] = src_np[:,-1]
    tgt_point_cloud["intensity"] = tgt_np[:,-1]

    #src_point_cloud.plot(eye_dome_lighting=True,render_points_as_spheres=True,color="cyan")
    #tgt_point_cloud.plot(eye_dome_lighting=True,render_points_as_spheres=True,color="red")

    plotter = pv.Plotter()
    plotter.add_mesh(src_point_cloud,render_points_as_spheres=True,color="cyan") #,eye_dome_lighting=True)
    plotter.add_mesh(tgt_point_cloud,render_points_as_spheres=True,color="red") #,eye_dome_lighting=True)
    plotter.show()




feature_match.show(run=True)
