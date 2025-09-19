"""
"""

from pathlib import Path

from magicgui import magicgui
import numpy as np
import napari
from stl import mesh
from napari_cool_tools_vol_proc._projection_tools_funcs import projection, ProjectionType, ProjectionDir
from batch_tools_utils import (
    scan_angle_fit_func,
    spherical_to_cartesian,
    spherical_to_cartesian_corrected,
)

####################################
from napari.layers import Points
@magicgui(
    call_button="Update Point Position",
    x={"widget_type": "FloatSlider", "min": -10, "max": 10, "step": 0.0001},
    y={"widget_type": "FloatSlider", "min": -10, "max": 10, "step": 0.0001},
    z={"widget_type": "FloatSlider", "min": -10, "max": 10, "step": 0.0001},
    angle={'widget_type': 'FloatSlider', 'min': 0, 'max': 360, 'step': 0.0001, 'value': 0}
)
def update_point_position(points_layer: Points, x: float = 0, y: float = 0, z: float = 0, angle: float = 0):
    if points_layer is not None:
        # get initial position
        init_position = update_point_position.init_position
        # Create new data array with updated coordinates
        new_position = np.array([[x, y, z]]) + init_position
        # Update the points layer data
        points_layer.data = new_position

        theta = np.radians(angle)

        rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)], # Row 0 (Z)
        [0, 1, 0],                         # Row 1 (Y is fixed, factor is 1)
        [-np.sin(theta), 0, np.cos(theta)] # Row 2 (X)
        ])

        # Apply the rotation matrix to the layer's affine transform
        points_layer.rotate = rotation_matrix


####################################


dots_scan_data_path = Path(r"E:\_Curve_Correction_Measurements\Yakub_Model_Eye_Dots\16_39_42_unp_std_std_vol_SNL.prof")
dots_CAD_stl_path = Path(r"E:\_Curve_Correction_Measurements\Yakub_Model_Eye_Dots\Model Eye_bottom_template_19mmDots_2_ToSlice.STL")
imaging_range = 12. #12.
refractive_index = 1.33
pivot_point = 19.2
scan_angle = 100 #100 #105 #140
imaging_motor_position = 74.5
reference_motor_position = 70.0

imaging_range /= refractive_index
base_padding = pivot_point - imaging_range
reference_arm_shift = imaging_motor_position - reference_motor_position # doing final position - initial position allows you to add this component
reference_arm_shift = (reference_arm_shift * 0.5) / refractive_index

padding = base_padding + reference_arm_shift

def main()->None:
    viewer = napari.Viewer(show=False)
    viewer.open(dots_scan_data_path,plugin="napari-cool-tools-io")
    initial_layer = viewer.layers[-1]
    data = initial_layer.data

    viewer.layers.remove(initial_layer)

    pixel_spacing = imaging_range / data.shape[1]
    padding_pixel = int(padding / pixel_spacing)

    #################################################################################################
    # get cad data
    cad_mesh = mesh.Mesh.from_file(dots_CAD_stl_path)
    #cad_points = np.column_stack([cad_mesh.v0,cad_mesh.v1,cad_mesh.v2])
    all_verts = cad_mesh.vectors.reshape(-1,3)
    print(f"all verts shape: {all_verts.shape}\n")
    cad_points = np.unique(all_verts,axis=0)
    cad_points[:,1] = cad_points[:,1]*-1
    

    min_coords = cad_points.min(axis=0)
    max_coords = cad_points.max(axis=0)

    #cad_points = cad_points / pixel_spacing
    
    dimensions = max_coords - min_coords

    dimensions = max_coords - min_coords
    
    length = dimensions[0]
    height = dimensions[1]
    width = dimensions[2]
    cad_com = cad_points.mean(axis=0)
    
    print(f"Dimensions (CAD): Length={length}, Height={height}, Width={width}, COM={cad_com}\n")

    #cad_points = np.around(np.unique(cad_mesh.vectors.reshape(-1, 3), axis=0), decimals=6)
    #print(f"cad points:\n{cad_points}\ncad points shape: {cad_points.shape}\n")

    ################################################################################################
    # Get scan data
    slow_axis_idxs = np.arange(data.shape[0])
    fast_axis_idxs = np.arange(data.shape[2])

    slow_axis_coords,fast_axis_coords = np.meshgrid(slow_axis_idxs,fast_axis_idxs,sparse=False)#np.ogrid[:data.shape[0],:data.shape[2]]
    axial_coords = projection(data,projection_type=ProjectionType.ARGMAX.value,axis=ProjectionDir.EN_FACE.value)
    #print(f"slow axis:\n{slow_axis_coords}\nfast axis:\n{fast_axis_coords}\naxial_axis:\n{axial_coords}\n")

    points_3D = np.column_stack([slow_axis_coords.flatten(),axial_coords[slow_axis_coords,fast_axis_coords].flatten(),fast_axis_coords.flatten()])
    #print(f"{points_3D.shape}\n")

    #viewer.add_image(axial_coords)
    #viewer.add_points(points_3D*pixel_spacing,size=0.1,face_color="blue",border_color="cyan",blending="additive")

    curv_points_3D = spherical_to_cartesian(points_3D,input_shape=data.shape,scan_angle=100,padding_pixel=padding_pixel)
    curv_points_3D_corrected = spherical_to_cartesian_corrected(points_3D,input_shape=data.shape,angle_func=scan_angle_fit_func,padding_pixel=padding_pixel)

    #curv_points_3D_corrected_len = curv_points_3D_corrected*pixel_spacing/refractive_index # TODO figure this out
    curv_points_3D_corrected_len = curv_points_3D_corrected*pixel_spacing #/refractive_index # TODO figure this out

    min_curv_coords = curv_points_3D_corrected_len.min(axis=0)
    max_curv_coords = curv_points_3D_corrected_len.max(axis=0)

    #cad_points = cad_points / pixel_spacing
    
    curv_dimensions = max_curv_coords - min_curv_coords

    curv_dimensions = max_curv_coords - min_curv_coords
    
    curv_length = curv_dimensions[0]
    curv_height = curv_dimensions[1]
    curv_width = curv_dimensions[2]
    curv_com = curv_points_3D_corrected_len.mean(axis=0)
    
    print(f"Dimensions (scan): Length={curv_length}, Height={curv_height}, Width={curv_width}, COM={curv_com}\n")

    # align centers of mass for rough lineup
    new_cad_offset = curv_com - cad_com
    new_cad_points = cad_points + new_cad_offset
    new_cad_com = new_cad_points.mean(axis=0)

    #viewer.add_points(curv_points_3D,size=0.1,face_color="green",border_color="yellow",blending="additive")
    #viewer.add_points(curv_points_3D_corrected,size=0.1,face_color="red",border_color="orange",blending="additive")
    viewer.add_points(curv_points_3D_corrected_len,size=0.01,face_color="red",border_color="orange",blending="additive")
    viewer.add_points(curv_com[None,:],size=0.5,face_color="purple",border_color="yellow",blending="additive",name="curv COM")
    viewer.add_points(cad_points,size=0.1,face_color="yellow",border_color="purple",blending="additive")
    viewer.add_points(cad_com[None,:],size=0.5,face_color="green",border_color="blue",blending="additive",name="cad COM")
    viewer.add_points(new_cad_points,size=0.1,face_color="green",border_color="blue",blending="additive")
    viewer.add_points(new_cad_com[None,:],size=0.5,face_color="yellow",border_color="purple",blending="additive",name="new cad COM")


    # Add the magicgui widget to the viewer
    viewer.window.add_dock_widget(update_point_position)
    update_point_position.init_position = new_cad_points

    # Connect the widget to run automatically when sliders change
    update_point_position.x.changed.connect(lambda: update_point_position(points_layer=viewer.layers['new_cad_points'], x=update_point_position.x.value, y=update_point_position.y.value, z=update_point_position.z.value,angle=update_point_position.angle.value))
    update_point_position.y.changed.connect(lambda: update_point_position(points_layer=viewer.layers['new_cad_points'], x=update_point_position.x.value, y=update_point_position.y.value, z=update_point_position.z.value,angle=update_point_position.angle.value))
    update_point_position.z.changed.connect(lambda: update_point_position(points_layer=viewer.layers['new_cad_points'], x=update_point_position.x.value, y=update_point_position.y.value, z=update_point_position.z.value,angle=update_point_position.angle.value))
    update_point_position.angle.changed.connect(lambda: update_point_position(points_layer=viewer.layers['new_cad_points'], x=update_point_position.x.value, y=update_point_position.y.value, z=update_point_position.z.value,angle=update_point_position.angle.value))
    

    viewer.show()
    napari.run()

if __name__ == "__main__":
    main()