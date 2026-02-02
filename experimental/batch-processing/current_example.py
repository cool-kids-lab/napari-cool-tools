""" """

from dataclasses import dataclass
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import grad
import napari
import numpy as np
import open3d as o3d
from scipy.optimize import fmin_bfgs
import torch
import torch.nn.functional as F

from current_dev import create_patch_processor, generate_patch_indices
from curve_correct_utils import (
    CurvCorrectSettings,
    equidistant_loss,
    generate_noisy_ellipsoid_sample_data,
    get_incidence_angle_torch,
    get_pixel_spacing_and_padding,
)
from napari_cool_tools_io._npz_reader import npz_file_reader
from napari_cool_tools_registration._fitting_funcs import (
    extract_surfaces_and_curve_correct_coordinates,
    sphere_fit_thick_map_corrected_v3,
)
from napari_cool_tools_segmentation._label_cleaning_funcs_v2 import (
    generate_elliptical_mask,
)

logging.basicConfig(
    level=logging.INFO,  # Capture INFO and above
    format="%(levelname)s: %(message)s",
)

jax.config.update("jax_enable_x64", True)


sample_data_path = Path(
    r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output\Clean_Labels_Topo_Maps\08983292-2024_06_19-14_03_16_ret_chor_seg_clean.npz"
    # r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output\Clean_Labels_Topo_Maps\08829838-2023_06_14-13_52_55_ret_chor_seg_clean.npz"
)

def generate_center_point_guess(
    curve_corrected_points: np.ndarray,
    cc_settings: CurvCorrectSettings,
    pixel_spacing: float,
):
    """"""

    # maximum distance down axial axis
    max_axial = curve_corrected_points[:, 1].max()
    # generate inital guess for center of ellipsoid
    center_guess_slow = curve_corrected_points[:, 0].mean()
    center_guess_fast = curve_corrected_points[:, 2].mean()
    center_guess_axial = max_axial - ((cc_settings.pivot_point / 2.0) / pixel_spacing)

    return np.array([center_guess_slow, center_guess_axial, center_guess_fast])

def sphere_fit_points(points_to_fit: np.ndarray, center_point_guess: np.ndarray):
    """"""
    # perform sphere fitting on data
    (
        center_point_result,
        min_variance,
        gradient,
        hessian_matrix,
        function_calls,
        gradient_calls,
        warning_flag,
    ) = fmin_bfgs(
        f=equidistant_loss,
        x0=center_point_guess.astype("float64"),
        fprime=grad(equidistant_loss),
        norm=2.0,
        args=(points_to_fit.astype("float64"),),
        # gtol=1e-17,
        maxiter=None,
        full_output=True,
        disp=False,  # True False
        retall=False,
        callback=None,
    )

    return center_point_result, min_variance

def generate_angle_of_incidence_map(points_to_map:np.ndarray,angles_of_incidence:np.ndarray,map_shape:tuple[int]):
    angle_of_incidence_map = np.zeros(map_shape, dtype=np.float32)
    angle_of_incidence_map[points_to_map[:, 0], points_to_map[:, 2]] = (
        angles_of_incidence.to(torch.float32).numpy()
    )
    return angle_of_incidence_map


def test_function():
    viewer = napari.Viewer(show=False)

    ray_density = 50

    # initiallize curve correction settings
    cc_settings = CurvCorrectSettings(
        imaging_range=12.0,  # 6., #12.,
        imaging_motor_position_delta=6.0,  # 0., #6.,
    )

    # load sample data and metadata
    sample_data, attributes, layer_type = npz_file_reader(
        str(sample_data_path), return_layer=True, verbose=True
    )[0]

    # process metadata
    if "metadata" in attributes:
        if "motor_position" in attributes["metadata"]:
            imaging_motor_position = attributes["metadata"]["motor_position"]
            cc_settings.imaging_motor_position = imaging_motor_position / 1000
        else:
            logging.info("Missing metadata skipping.")
            return
    else:
        logging.info("Missing motor_position in metadata skipping.")
        return

    # get pixel spacing and padding for the volume
    pixel_spacing, padding_pixel = get_pixel_spacing_and_padding(
        cc_settings=cc_settings, axial_data_shape=sample_data.shape[1], verbose=False
    )

    # extract surface points from segmentation data and curvecorrect samples
    retina = sample_data == 1
    if retina.sum() > 0:
        (
            cc_micro_thickness_map,
            cc_pixel_thickness_map,
            curv_ret_points,
            curv_rpe_points,
            retina_points,
            rpe_points,
            cc_retina_nn_points,
            raw_micro_thickness_map,
            raw_pixel_thickness_map,
        ) = extract_surfaces_and_curve_correct_coordinates(
            retina,
            pixel_spacing=pixel_spacing,
            padding_pixel=padding_pixel,
            refractive_index=cc_settings.refractive_index,
        )

        # (
        #     thickness_map,
        #     retina_points,
        #     rpe_points,
        #     curv_ret_points,
        #     curv_rpe_points,
        #     raw_pixel_thickness_map,
        #     pixel_thickness_map,
        #     raw_micron_thickness_map,
        # ) = sphere_fit_thick_map_corrected_v3(
        #     retina,
        #     pixel_spacing=pixel_spacing,
        #     padding_pixel=padding_pixel,
        #     refractive_index=cc_settings.refractive_index,
        # )
    else:
        logging.info("Label data is missing skipping.")
        return
    
    nearest_neighbor_dir = curv_ret_points-cc_retina_nn_points
    #nearest_neighbor_dir = curv_rpe_points-cc_rpe_nn_points

    
    points_of_interest = rpe_points
    curv_points_of_interest = curv_rpe_points #TODO change this back this was for Beth's image
    # curv_points_of_interest = curv_ret_points

    # curv_other_points = curv_ret_points
    curv_other_points = curv_rpe_points

    logging.info(f"curved correcte points shape: {curv_points_of_interest.shape}")

    nn_angles_of_incidence = get_incidence_angle_torch(
        torch.as_tensor(curv_points_of_interest, dtype=torch.bfloat16),
        torch.as_tensor(nearest_neighbor_dir, dtype=torch.bfloat16),
        use_degrees=True,
    )

    logging.info(
        f"Angles shape: {nn_angles_of_incidence.shape}, min/max: {nn_angles_of_incidence.min()}/{nn_angles_of_incidence.max()}"
    )

    nn_angle_of_incidence_map = generate_angle_of_incidence_map(points_of_interest,nn_angles_of_incidence,(840,800))

    # nn_start = curv_points_of_interest
    # nn_end = nearest_neighbor_dir - curv_points_of_interest

    nn_norms = np.zeros((len(curv_points_of_interest),2,3))
    nn_norms[:,0,:] = curv_points_of_interest #nn_start
    nn_norms[:,1,:] = nearest_neighbor_dir #nn_end #nearest_neighbor_dir-curv_points_of_interest

    # Sphere fit data
    center_point_guess = generate_center_point_guess(
        curv_points_of_interest, cc_settings=cc_settings, pixel_spacing=pixel_spacing
    )
    center_point, variance = sphere_fit_points(curv_points_of_interest, center_point_guess=center_point_guess)
    direction_raw = center_point-curv_points_of_interest
    center_based_normals = direction_raw/np.linalg.norm(direction_raw,axis=1,keepdims=True)

    center_point_um = center_point*pixel_spacing
    variance_um = variance*pixel_spacing
    std_um = np.sqrt(variance_um)
    radius = np.sqrt(np.sum((curv_points_of_interest - center_point)**2,axis=1)).mean()

    logging.info(
        f"Center: {center_point_um}, Radius: {radius} (pixels), {radius*pixel_spacing} (um), std_um {std_um}"
    )

    ellipsoid_points = generate_noisy_ellipsoid_sample_data(tuple(center_point),radius=radius,add_noise=False)

    center_norms = np.zeros((len(curv_points_of_interest),2,3))
    center_norms[:,0,:] = curv_points_of_interest
    center_norms[:,1,:] = direction_raw

    center_angles_of_incidence = get_incidence_angle_torch(
        torch.as_tensor(curv_points_of_interest, dtype=torch.bfloat16),
        torch.as_tensor(direction_raw, dtype=torch.bfloat16),
        use_degrees=True,
    )

    logging.info(
        f"Angles shape: {center_angles_of_incidence.shape}, min/max: {center_angles_of_incidence.min()}/{center_angles_of_incidence.max()}"
    )

    center_angle_of_incidence_map = generate_angle_of_incidence_map(points_of_interest,center_angles_of_incidence,(840,800))

    # open3d mesh generation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(curv_points_of_interest)

    # estimate normals TODO investigate https://github.com/jsnln/ParametricGaussRecon & https://github.com/jsnln/WNNC more advanced point cloud surface reconstruction
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=6))
    pcd.orient_normals_consistent_tangent_plane(k=14)
    estimated_normals = np.asarray(pcd.normals)
    logging.info(f"curved corrected estimated normals shape: {estimated_normals.shape}")

    esti_norms = np.zeros((len(curv_rpe_points),2,3))
    esti_norms[:,0,:] = curv_points_of_interest
    esti_norms[:,1,:] = estimated_normals-curv_points_of_interest

    # bet rays from pivot point
    # get pivot points for labeling
    pivot_point_vector = np.array([0.0, 0.0, 0.0])
    ray_vectors = curv_points_of_interest #rpe_points  # retina_points

    angles_of_incidence = get_incidence_angle_torch(
        torch.as_tensor(ray_vectors, dtype=torch.bfloat16),
        torch.as_tensor(estimated_normals, dtype=torch.bfloat16),
        use_degrees=True,
    )

    logging.info(
        f"Angles shape: {angles_of_incidence.shape}, min/max: {angles_of_incidence.min()}/{angles_of_incidence.max()}"
    )

    angle_of_incidence_map = generate_angle_of_incidence_map(points_of_interest,angles_of_incidence,(840,800))
    sphere_v_mean_diff = center_angle_of_incidence_map-angle_of_incidence_map
    mean_v_nn_diff = angle_of_incidence_map-nn_angle_of_incidence_map

    #### Add stuff to the viewer ####
    viewer.add_points(
        curv_other_points,
        size=6.0,
        border_color="green",
        face_color="green",
        blending="translucent_no_depth",
    )
    viewer.add_points(
        curv_points_of_interest,
        size=1.0,
        border_color="white",
        face_color="red",
        blending="translucent_no_depth",
    )
    viewer.add_vectors(
        nn_norms[::ray_density,:],
        edge_width=0.4,
        length=1.0,
        edge_color="magenta",
        vector_style="line",
        name="nn_norms",
    )
    viewer.add_points(
        ellipsoid_points,
        size=40,
        border_color="purple",
        face_color="yellow",
        blending="translucent_no_depth",
    )
    viewer.add_vectors(
        esti_norms[::ray_density,:],
        edge_width=0.1,
        length=0.05,
        edge_color="yellow",
        vector_style="line",
        name="estimated_norms",
    )
    viewer.add_vectors(
        center_norms[::ray_density,:],
        edge_width=0.1,
        length=1.0,
        edge_color="cyan",
        vector_style="line",
        name="center_norms",
    )

    viewer.add_image(cc_micro_thickness_map>0,name="mask")
    viewer.add_image(sphere_v_mean_diff)
    viewer.add_image(mean_v_nn_diff)
    viewer.add_image(cc_micro_thickness_map)
    viewer.add_image(nn_angle_of_incidence_map, name="nn_angle_of_incedence_map")
    viewer.add_image(center_angle_of_incidence_map, name="center_angle_of_incedence_map")
    viewer.add_image(angle_of_incidence_map, name="angle_of_incedence_map")

    logging.info(f"Spherical-Mean_Norm: {sphere_v_mean_diff.mean()}/{sphere_v_mean_diff.max()} ({sphere_v_mean_diff.std()})")
    logging.info(f"Mean_Norm-NN: {mean_v_nn_diff.mean()}/{mean_v_nn_diff.max()} ({mean_v_nn_diff.std()})")

    ##### Visualization stuff

    # 4. Run Poisson Reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    densities = np.asarray(densities)
    # Using a quantile (e.g., bottom 1% to 5%) is a common starting point
    vertices_to_remove = densities < np.quantile(densities, 0.01)

    # Remove the vertices and clean the mesh
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.remove_unreferenced_vertices()

    # 5. for o3d
    mesh.compute_vertex_normals()
    mesh.orient_triangles()
    mesh.vertex_colors = o3d.utility.Vector3dVector([])

    #6 flip normals and reverse triangle winding
    # mesh.vertex_normals = o3d.utility.Vector3dVector(-np.asarray(mesh.vertex_normals))
    # mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, [0, 2, 1]])

    # 7. mask mesh

    # Create a temporary PointCloud from the mesh vertices
    # temp_pcd = o3d.geometry.PointCloud()
    # temp_pcd.points = mesh.vertices

    # dist_to_og_sample_data = temp_pcd.compute_point_cloud_distance(pcd)
    # dist_to_og_sample_data = np.asarray(dist_to_og_sample_data)

    # threshold_to_pcd = 0.5
    # vert_mask = dist_to_og_sample_data > threshold_to_pcd
    # print(f"vert_mask sum: {vert_mask.sum()}")
    # # vertices = np.asarray(mesh.vertices)

    # # def get_ellipsoid_mask(vertices, center:tuple[float,float,float]=(0.,0.,0.), radii:tuple[float,float,float] = (1.,1.,1.)):
    # #     """"""
    # #     dist = (
    # #         ((vertices[:, 0] - center[0])**2 / radii[0]**2) +
    # #         ((vertices[:, 1] - center[1])**2 / radii[1]**2) +
    # #         ((vertices[:, 2] - center[2])**2 / radii[2]**2)
    # #     )
    # #     return dist <= 1.0
    
    # # vert_mask = get_ellipsoid_mask(vertices,center=(0.,0.,0.),radii=(radius,radius,(800/840)*radius))
    # # vert_mask = get_ellipsoid_mask(vertices,center=tuple(center_point),radii=(radius,radius,(800/840)*radius))

    # mesh.remove_vertices_by_mask(~vert_mask)

    # Standard cleanup after modification
    # mesh.remove_unreferenced_vertices()
    # mesh.compute_vertex_normals()
    # mesh.orient_triangles()
    mesh.vertex_colors = o3d.utility.Vector3dVector([])

    # 8. setup material
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    # mat.shader = "defaultUnlit"
    mat.base_color = [1.0, 0.1, 0.1, 1.0] # RGBA
    mat.base_roughness = 0.2
    mat.base_metallic = 0.0

    o3d.visualization.draw([{'name':'beth_figure','geometry':mesh,'material':mat}])
    # o3d.visualization.draw_geometries([mesh])

    # vertices = np.asarray(mesh.vertices)
    # faces = np.asarray(mesh.triangles)

    # # viewer.add_surface((vertices,faces),name="Reconstructed Mesh",colormap='gray')

    viewer.show()
    napari.run()


test_function()
