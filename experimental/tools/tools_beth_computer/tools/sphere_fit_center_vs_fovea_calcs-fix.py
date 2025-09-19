"""
"""

from pathlib import Path

from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax import grad
from jax import random
import numpy as np
from scipy.optimize import fmin_bfgs
from scipy.spatial import KDTree
import napari
from napari.utils.notifications import show_info
from magicgui.widgets import Container, FileEdit, LineEdit, PushButton, Label
from magicgui import magicgui

from batch_tools_utils import (
    generate_height_map_from_3D_points,
    get_motor_pos_from_xml,
    get_nearest_neigbors_to_target,
    load_bits_labels_v2,
    scan_angle_fit_func,
    sphere_fit_thick_map,
    sphere_fit_thick_map_corrected,
    spherical_to_cartesian,
    spherical_to_cartesian_corrected,
)

jax.config.update("jax_enable_x64", True)

# def get_y_coordinate_value(values:tuple(float),indices:tuple(int),coordinates:np.ndarray):
#     """"""
#     total_indices = set(np.arange(len(coordinates.shape[1])))
#     missing_indices = set(total_indices).difference(indices)

#     masks = []
#     for idx in indices:
#         mask = coordinates[:,idx] == 

def get_y_coordinate_from_x_and_z_values(x_value,z_value,coordinates):
    """
    """
    x_mask = coordinates[:,0] == x_value
    z_mask = coordinates[:,2] == z_value
    target_idx = (x_mask & z_mask).nonzero()[0]
    return coordinates[target_idx,1][0]

def get_y_coordinate_from_x_and_z_values_v2(coordinates):
    """
    """
    argmin_x = np.abs(coordinates[:,0]).argmin()
    argmin_z = np.abs(coordinates[:,2]).argmin()
    x_mask = coordinates[:,0] == coordinates[argmin_x,0]
    z_mask = coordinates[:,2] == coordinates[argmin_z,2]
    target_idx = (x_mask & z_mask).nonzero()[0]
    #target_idx = coordinates[argmin_x,:,argmin_z]
    return coordinates[target_idx,1][0]

def get_x_and_z_coordinates_from_y_value(y_value,coordinates):
    """
    """
    target_idx = (coordinates[:,1] == y_value).nonzero()[0]  
    return (coordinates[target_idx,0][0],coordinates[target_idx,2][0])

def generate_noisy_ellipsoid_sample_data(center:tuple[int,int,int]=(0,0,0),semi_axes:tuple[float,float,float]=(1.0,1.0,800/840),radius:float=1.0,theta_samples:int=20,seed:int=42,add_noise:bool=True):
    """
    modified from https://jekel.me/2021/A-better-way-to-fit-Ellipsoids/
    """
    # create noise
    if add_noise:
        np.random.seed(seed)
        noise = np.random.normal(size=(theta_samples*theta_samples), loc=0, scale=1e-2)
    else:
        noise = 0
        #noise = np.zeros((theta_samples*theta_samples,))

    # define u,v space which equate to theta and phi in spherical coordinates
    u = np.linspace(0.,np.pi*2,theta_samples)
    v = np.linspace(0.,np.pi, theta_samples)
    u,v = np.meshgrid(u,v,sparse=True)
    a = semi_axes[0] #1.0
    b = semi_axes[1] #0.5,
    c = semi_axes[2] #800/840

    # calculate cartesian coordinates from spherical
    x = a*np.cos(u)*np.sin(v)*radius
    y = b*np.cos(v)*radius
    z = c*np.sin(u)*np.sin(v)*radius

    x = x.flatten() + noise
    y = np.repeat(y.flatten(),theta_samples) + noise
    z = z.flatten() + noise

    x = x+center[0]
    y = y+center[1]
    z = z+center[2]

    return np.column_stack((x,y,z))

def generate_ellipsoid_samples(theta_samples:int,phi_samples:int,center_x:float,center_y:float,center_z:float,param_a:float,param_b:float,param_c:float):
    """
    modified from https://jekel.me/2021/A-better-way-to-fit-Ellipsoids/
    """
    u = np.linspace(0.,np.pi*2,theta_samples)
    v = np.linspace(0.,np.pi,phi_samples)
    u,v = np.meshgrid(u,v,sparse=True) # can this be done with ogrid?

    x = param_a*np.cos(u)*np.sin(v)
    y = param_c*np.cos(v)
    z = param_b*np.sin(u)*np.sin(v)
    
    x = x.flatten() + center_x
    y = np.repeat(y.flatten(),theta_samples,axis=0) + center_y
    z = z.flatten() + center_z

    return np.column_stack((x,y,z))

def generate_ellipsoid_paraneter_guesses_old(center_slow:float,center_fast:float,center_axial_range:tuple[float,float]=(15.0,17.0)):
    """
    modified from https://jekel.me/2021/A-better-way-to-fit-Ellipsoids/
    """
    gamma_guess = np.random.random(6)
    center_axial = np.random.uniform(center_axial_range[0],center_axial_range[1])#*1000
    gamma_guess[0] = center_slow
    gamma_guess[1] = center_axial
    gamma_guess[2] = center_fast
    gamma_guess = jnp.array(gamma_guess)
    return gamma_guess

def generate_ellipsoid_paraneter_guesses(center_slow:float,center_axial:float,center_fast:float):
    """
    modified from https://jekel.me/2021/A-better-way-to-fit-Ellipsoids/
    """
    gamma_guess = np.random.random(6)
    #center_axial = np.random.uniform(center_axial_range[0],center_axial_range[1])#*1000
    gamma_guess[0] = center_slow
    gamma_guess[1] = center_axial
    gamma_guess[2] = center_fast
    gamma_guess = jnp.array(gamma_guess)
    return gamma_guess

def predict(gamma,x:np.ndarray,y:np.ndarray,z:np.ndarray):
    """
    modified from https://jekel.me/2021/A-better-way-to-fit-Ellipsoids/
    """
    # compute f hat
    x0 = gamma[0]
    y0 = gamma[1]
    z0 = gamma[2]
    a2 = gamma[3]**2
    b2 = gamma[4]**2
    c2 = gamma[5]**2
    zeta0 = (x - x0)**2 / a2
    zeta1 = (y - y0)**2 / b2
    zeta2 = (z - z0)**2 / c2
    return zeta0 + zeta1 + zeta2

def loss(gamma_guess,x:np.ndarray,y:np.ndarray,z:np.ndarray):
    """
    modified from https://jekel.me/2021/A-better-way-to-fit-Ellipsoids/
    """
    # compute mean squared error
    pred = predict(gamma_guess,x,y,z)
    target = jnp.ones_like(pred)
    mse = jnp.square(pred-target).mean()
    return mse

def equidistant_loss(center_point_3D:np.ndarray, points_3D:np.ndarray):
    """
    """
    distances = jnp.sqrt(jnp.sum((points_3D.astype("float64") - center_point_3D.astype("float64"))**2,axis=1))
    # print(f"distances: {distances}\n")
    # mean_distance = jnp.mean(distances)
    # print(f"mean_distance: {mean_distance}\n")
    # std = jnp.std(distances)
    # print(f"std2:{std}\n")
    return jnp.var(distances) #jnp.std(distances)
    #return jnp.sqrt(jnp.sum((distances - mean_distance)**2))

def distribution_to_ellipsoid_loss(ellipsoid_params:np.ndarray,points_3D:np.ndarray):
    """
    """
    center = ellipsoid_params[:3]
    coefficients = ellipsoid_params[3:]
    radius = np.sqrt(np.sum((points_3D - center)**2,axis=1)).mean()
    ellipsoid_points = generate_noisy_ellipsoid_sample_data(tuple(center),semi_axes=tuple(coefficients),radius=radius,theta_samples=840,add_noise=False)
    tree = KDTree(ellipsoid_points)
    distances,indices = tree.query(points_3D,k=1)
    return jnp.var(distances)

def generate_sphere_fit(label_data:np.ndarray,imaging_motor_position:float,pivot_point:float,scan_angle:float,viewer:napari.Viewer):
    """
    """
    # check that retina labels exist
    if label_data.sum() > 0:
        pass
    else:
        return 

    # get sphere_fit_data
    (
        thickness_map, retina_coords, rpe_coords, curv_ret_coords, curv_rpe_coords, raw_pixel_thickness_map, pixel_thickness_map
    #) = sphere_fit_thick_map(label_data,imaging_motor_position=imaging_motor_position,pivot_point=pivot_point,scan_angle=scan_angle)
    ) = sphere_fit_thick_map_corrected(label_data,imaging_motor_position=imaging_motor_position,pivot_point=pivot_point,scan_angle=scan_angle)

    #viewer = napari.Viewer(show=False)
    # viewer.add_points(curv_ret_coords,face_color="red",border_color="orange",size=4)
    # viewer.add_points(curv_rpe_coords,face_color="cyan",border_color="blue",size=4)

    return rpe_coords,curv_ret_coords,curv_rpe_coords,thickness_map

#### TODO Review Improve ####
def create_cone_points(tip_pos, height, angle_degrees, num_segments=20):
    """
    Generates points for a cone with its tip at tip_pos, 
    oriented along the negative z-axis by default.
    The cone angle is the half-angle of the cone.
    
    For custom angles and rotations, manual rotation matrix application is needed.
    """
    import math
    half_angle_rad = math.radians(angle_degrees / 2)
    radius = height * math.tan(half_angle_rad)
    #base_center = tip_pos + np.array([0, 0, height]) # Base is 'above' the tip in z-direction
    base_center = tip_pos + np.array([0, height, 0]) # Base is 'above' the tip in z-direction
    
    # Generate points around the base circle
    theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    x = base_center[0] + radius * np.cos(theta)
    # y = base_center[1] + radius * np.sin(theta)
    # z = np.full(num_segments, base_center[2])
    y = np.full(num_segments, base_center[1])
    z = base_center[2] + radius * np.sin(theta)
    
    #base_points = np.stack([z, y, x], axis=-1) # Napari uses (z, y, x) order
    base_points = np.stack([x, y, z], axis=1) # Napari uses (z, y, x) order
    
    # The tip point (also in z, y, x order)
    #tip_point = np.array([tip_pos[2], tip_pos[1], tip_pos[0]])
    tip_point = tip_pos.copy()

    # Combine base points with the tip to define the mesh surface
    # This requires a mesh/surface layer, not just a shapes layer
    # For a simple representation, you could use points layer to show vertices
    
    all_points = np.vstack([base_points, tip_point])
    return all_points, base_points

if __name__ == "__main__":

    data_path = Path(r"E:\_Beth_Thickness_Calculations\processed2\UNPs_08977490-2024_05_15-14_53_55_processed_ret_chor_seg.npz")
    xml_path = Path(r"E:\_Beth_Thickness_Calculations\All xmls\08977490-2024_05_15-14_53_55.xml")
    # data_path = Path(r"F:\_Complex_Conjugate_Problem\6_mm_test\processed2\08983292-2024_05_01-14_37_16_ret_chor_seg.npz")
    # xml_path = Path(r"F:\_Complex_Conjugate_Problem\_Complex_Conjugate_Problem_08983292-2024_05_01-14_37_16_processed.xml")

    label_data,metadata,layer_type = load_bits_labels_v2(data_path)
    reference_motor_position = 85.0
    pivot_point=19.2 #15.1 #19.2
    
    #pivot_point=  12.26 #19.2 #12.62 #15-2.38 #19 #16.5 - 2.38
    retina_data = label_data == 1

    # Get retina label data and xml motot position data
    imaging_motor_position = get_motor_pos_from_xml(xml_path=xml_path)
    #imaging_motor_position = 85.0
    print(f"imaging_motor_position: {imaging_motor_position}\n")
    if imaging_motor_position:
        imaging_motor_position =  imaging_motor_position/ 1000
        print(f"imaging_motor_position: {imaging_motor_position}\n")
    else:
        print("skipping...")

    initial_imaging_motor_position = imaging_motor_position

    use_min_sample:bool =  False #True #False

    dir = 1.
    learning_rate = 0.1
    decay = 0.9
    magnitude = 1.0
    iterations = 1 #10 #20 #10 #5 #1
    error = 1000000.
    min_std = 1000000.
    min_radius = 19.
    min_max_radius = 19.
    min_motor_position = initial_imaging_motor_position
    min_curv_rpe = None
    min_min_distort_curv_rpe = None
    min_curv_ret = None
    min_curv_fovea = None
    min_ppp_vector = None
    min_center_pt = None

    ratio_near_target = 0.5
    
    imaging_range = 12. #6. #12.
    refractive_index = 1.33
    #pivot_point = pivot_point / refractive_index
    scan_angle = 100 #100 #105 #140

    imaging_range /= refractive_index
    pixel_spacing = imaging_range / retina_data.shape[1] #z_shape
    print(f"pixel_spacing: {pixel_spacing}\n")

    print(f"pivot point: {pivot_point}\n")
    print(f"imaging range: {imaging_range}\n")
    base_padding = pivot_point - imaging_range
    print(f"base padding (pivot point - imaging range): {base_padding}\n")

    reference_arm_shift = imaging_motor_position - reference_motor_position # doing final position - initial position allows you to add this component
    reference_arm_shift = (reference_arm_shift * 0.5) / refractive_index
    #reference_arm_shift = reference_arm_shift / refractive_index

    
    print(f"initial reference arm shift: {reference_arm_shift}\n")

    padding = base_padding + reference_arm_shift
    print(f" initial padding: {padding}\n")

    for idx in tqdm(range(iterations),desc="Iterations"):

        # # Adjust reference arm shift
        # #reference_arm_shift = reference_motor_position - imaging_motor_position
        # reference_arm_shift = imaging_motor_position - reference_motor_position # doing final position - initial position allows you to add this component
        
        # reference_arm_shift = (reference_arm_shift * 0.5) / refractive_index
        # #reference_arm_shift = reference_arm_shift/refractive_index
        # print(f"reference arm shift: {reference_arm_shift}\n")

        # padding = base_padding + reference_arm_shift
        # print(f"padding: {padding}\n")
        # if padding < 0:
        #     dir = 1.
        #     magnitude = decay * magnitude
        #     delta = learning_rate*initial_imaging_motor_position*dir*magnitude
        #     print(f"delta: {delta}\n")
        #     imaging_motor_position = delta+imaging_motor_position
        #     print(f"new imaging motor position: {imaging_motor_position}\n")
        #     continue

        padding_pixel = int(padding / pixel_spacing)
        
        # Create the widget and show it
        viewer = napari.Viewer(show=False)

        # generate sphere fit TODO
        (rpe_points, curv_ret_points, curv_rpe_points, thickness_map) = generate_sphere_fit(
        #(rpe_points, _, curv_rpe_points, thickness_map) = generate_sphere_fit(
            label_data=retina_data,
            imaging_motor_position=imaging_motor_position,
            pivot_point=pivot_point,
            scan_angle=scan_angle,
            viewer=viewer,
        )

        # TODO check
        # correct to air for curved points
        #curv_ret_points = curv_ret_points / refractive_index
        #curv_rpe_points = curv_rpe_points / refractive_index

        # get pivot points for labeling
        pivot_point_vector = np.array([0.,0.,0.])
        #rpe_points = viewer.layers["rpe_coords"].data
        #curv_rpe_points = viewer.layers["curv_rpe_coords"].data

        # for idx in range(curv_rpe_points.shape[1]):
        #     print(curv_rpe_points[:,idx].shape)

        # create height map and get axial coordinate for fovea
        height_map = generate_height_map_from_3D_points(rpe_points.astype(int),(retina_data.shape[0],retina_data.shape[2]))
        fovea_y = int(height_map[[460],[456]][0])
        #print(f"fovea y:{fovea_y}\n")
        #fovea_y = get_y_coordinate_from_x_and_z_values(460.,456.,curv_rpe_points)
        fovea_point = np.array([460.,fovea_y,456.])
        #print(f"fovea_point: {fovea_point}, shape: {fovea_point.shape}\n")

        # curve correct foveal point
        #curv_fovea_point = spherical_to_cartesian(fovea_point[None,:],retina_data.shape,scan_angle=scan_angle,padding_pixel=padding_pixel)
        
        curv_fovea_point = spherical_to_cartesian_corrected(fovea_point[None,:],retina_data.shape,angle_func=scan_angle_fit_func,padding_pixel=padding_pixel)
        # TODO check
        #curv_fovea_point = curv_fovea_point / refractive_index

        #print(f"curv_fovea_point: {curv_fovea_point}, shape: {curv_fovea_point.shape}\n")

        # print(f"curved rpe: {curv_rpe_points},type: {type(curv_rpe_points),curv_rpe_points.dtype}\n")
        # print(np.abs(curv_rpe_points[:,0]).min())
        # print(np.abs(curv_rpe_points[:,2]).min())
        # import sys
        # sys.exit(0)

        #ppp_axial = get_y_coordinate_from_x_and_z_values(0.,0.,curv_rpe_points)
        #print(ppp_axial)
        #projected_pivot_point_vector = np.array([0.,ppp_axial,0.])

        max_axial = curv_rpe_points[:,1].max()
        #print(max_axial)
        max_slow,max_fast = get_x_and_z_coordinates_from_y_value(max_axial,curv_rpe_points)
        #print(max_slow,max_fast)
        max_point = np.array([max_slow,max_axial,max_fast])

        # generate inital guess for center of ellipsoid
        center_guess_slow = curv_rpe_points[:,0].mean()
        center_guess_fast = curv_rpe_points[:,2].mean()
        #center_guess_axial = max_axial - ((16.5/2.0) / pixel_spacing)
        #center_guess_axial = max_axial - ((15.1/2.0) / pixel_spacing)
        center_guess_axial = max_axial - ((pivot_point/2.0) / pixel_spacing)
        center_point_guess = np.array([center_guess_slow,center_guess_axial,center_guess_fast])

        
        #print(f"curv_rpe_points: {curv_rpe_points}\n")
        print(f"center_point_guess: {center_point_guess}\n")

        #if use_min_sample:
        minimal_distortion_curv_rpe = get_nearest_neigbors_to_target(target_point=curv_fovea_point[None,:],point_distribution=curv_rpe_points,ratio=ratio_near_target)

        print(f"minimal_distortion_curv_rpe shape: {minimal_distortion_curv_rpe.shape}\n")

        # import sys
        # sys.exit(0)

        if not use_min_sample:
            rpe_point_input = curv_rpe_points
        else:
            rpe_point_input = minimal_distortion_curv_rpe


        output = fmin_bfgs(
            f=equidistant_loss,
            x0=center_point_guess.astype("float64"),
            fprime=grad(equidistant_loss),
            norm=2.0,
            args=(rpe_point_input.astype("float64"),),
            #gtol=1e-17,
            maxiter=None,
            full_output=True,
            disp=True,
            retall=False,
            callback=None,
        )

        print(f"output: {output}\n")
        new_error = jnp.sqrt(output[1])
        print(f"Old error: {error}, new error: {new_error}\n")

        if idx == 0:
            min_std = np.sqrt(output[1])
            min_center_pt = output[0]
            #min_radius = np.sqrt(np.sum((curv_rpe_points - output[0])**2,axis=1)).mean()
            #min_radius = np.sqrt(np.sum((minimal_distortion_curv_rpe - output[0])**2,axis=1)).mean()
            min_radius = np.sqrt(np.sum((rpe_point_input - output[0])**2,axis=1)).mean()
            min_max_radius = np.sqrt(np.sum((max_point-output[0])**2))
            min_curv_rpe = curv_rpe_points
            min_min_distort_curv_rpe = minimal_distortion_curv_rpe
            min_curv_ret = curv_ret_points
            min_curv_fovea = curv_fovea_point
            #min_ppp_vector = projected_pivot_point_vector
            min_motor_position = imaging_motor_position

        if new_error > error: # if new error is greater than old error change directions
            dir = dir * -1
            magnitude = decay * magnitude

        else:
            magnitude = decay * magnitude
            min_std = np.sqrt(output[1])
            min_center_pt = output[0]
            #min_radius = np.sqrt(np.sum((curv_rpe_points - output[0])**2,axis=1)).mean()
            #min_radius = np.sqrt(np.sum((minimal_distortion_curv_rpe - output[0])**2,axis=1)).mean()
            min_radius = np.sqrt(np.sum((rpe_point_input - output[0])**2,axis=1)).mean()
            min_max_radius = np.sqrt(np.sum((max_point-output[0])**2))
            min_curv_rpe = curv_rpe_points
            min_min_distort_curv_rpe = minimal_distortion_curv_rpe
            min_curv_ret = curv_ret_points
            min_curv_fovea = curv_fovea_point
            #min_ppp_vector = projected_pivot_point_vector
            min_motor_position = imaging_motor_position

        error = new_error #output[1]

        # calculate radius based on distance from center of fit sphere to fovea
        new_center_guess = output[0].copy()
        #radius_calc = np.sqrt(np.sum((curv_rpe_points - new_center_guess)**2,axis=1)).mean()
        #radius_calc = np.sqrt(np.sum((minimal_distortion_curv_rpe - new_center_guess)**2,axis=1)).mean()
        radius_calc = np.sqrt(np.sum((rpe_point_input - new_center_guess)**2,axis=1)).mean()
        radius_guess = np.sqrt(np.sum((curv_fovea_point-new_center_guess)**2))
        radius_guess_max = np.sqrt(np.sum((max_point-new_center_guess)**2))
        #radius_guess_proj = np.sqrt(np.sum((projected_pivot_point_vector-new_center_guess)**2))
        #print(f"curved rpe points: {curv_rpe_points}\n")
        #print(f"minus new center: {curv_rpe_points - new_center_guess}\n")
        print(f"radius calculation from fitting: {radius_calc} pixels, {radius_calc*pixel_spacing} mm\n")
        print(f"radius guess in pixels center to fovea: {radius_guess}, pixels {radius_guess*pixel_spacing} mm\n")
        print(f"radius guess in pixels center to max axial distance from pivot point: {radius_guess_max}, {radius_guess_max*pixel_spacing} mm\n")
        #print(f"radius guess in pixels center to projected pivot point: {radius_guess_proj}, {radius_guess_proj*pixel_spacing} mm\n")

        # # generate points for ellipsoid centered on the center of the fit sphere with radius based on foveal distance
        # #ellipsoid_points = generate_noisy_ellipsoid_sample_data(tuple(new_center_guess),(1.0,1.0,1.0),radius=radius_guess,add_noise=False)
        # ellipsoid_points = generate_noisy_ellipsoid_sample_data(tuple(new_center_guess),(1.0,1.0,1.0),radius=radius_calc,add_noise=False)

        # # calculate line along predicted optical axis
        # offset = np.array([0.,radius_guess,0.])
        # spherical_axis_line_data = [new_center_guess-offset,new_center_guess+offset]

        # # calulate vectors for closest point along fit shpere axis to pivot point on pupil plane
        # sphere_axis_vector = (new_center_guess+offset) - (new_center_guess-offset)
        # axis_start_to_pivot_point_vector = pivot_point - (new_center_guess-offset)
        # scalar_projection = np.dot(axis_start_to_pivot_point_vector,sphere_axis_vector) / np.dot(sphere_axis_vector,sphere_axis_vector)
        # closest_point_on_sphere_axis = (new_center_guess-offset) + scalar_projection * sphere_axis_vector

        # # calculate axial length and anterior chamber depth
        # axial_length = np.sqrt((sphere_axis_vector**2).sum()) * pixel_spacing
        # anterior_chamber_depth =  np.sqrt(((closest_point_on_sphere_axis - (new_center_guess-offset))**2).sum()) * pixel_spacing
        # #print(f"axial length: {axial_length}\nanterior chamber depth: {anterior_chamber_depth}\n")

        # # calculate visual axis segment
        # visual_axis_line_data = [closest_point_on_sphere_axis,curv_fovea_point.squeeze()]
        # #print(spherical_axis_line_data,visual_axis_line_data)

        # # generate cone points for scan angle
        # cone_points,cone_base_points = create_cone_points((pivot_point).squeeze(),height=radius_guess,angle_degrees=scan_angle,num_segments=200)
        # #print(f"cone_points: {cone_points}")

        # # distance from anterior sphere to pupil plane
        # anterior_sphere_to_pupil_plane = np.sqrt(np.sum((closest_point_on_sphere_axis-(new_center_guess-offset))**2))
        # print(f"anterior sphere to pupil plane distance: {anterior_sphere_to_pupil_plane} pixels, {anterior_sphere_to_pupil_plane*pixel_spacing}\n")

        # cone_line_segments = []
        # for base_point in cone_base_points:
        #     line_segment = np.array([pivot_point.squeeze(),base_point])
        #     cone_line_segments.append(line_segment)
        
        # base_point_size = 80

        # viewer.add_points(rpe_points,face_color="white",border_color="gray",size=4)
        # viewer.add_points(curv_ret_points,face_color="red",border_color="gray",size=4)
        # viewer.add_points(curv_rpe_points,face_color="white",border_color="gray",size=4)
        # viewer.add_image(thickness_map)

        # viewer.add_image(height_map)
        # viewer.add_points(pivot_point,size=base_point_size,face_color="green",border_color="yellow",blending="additive",name="pivot_point")
        # viewer.add_points(projected_pivot_point,size=base_point_size,face_color="red",border_color="purple",blending="additive",name="projected_pivot_point")
        # viewer.add_points(max_point,size=base_point_size,face_color="purple",border_color="yellow",blending="additive",name="max_point")
        # #viewer.add_points(fovea_point,size=40,face_color="purple",border_color="yellow",blending="additive",name="fovea_point")
        # viewer.add_points(curv_fovea_point,size=base_point_size,face_color="yellow",border_color="purple",blending="additive",name="curv_fovea_point")
        # viewer.add_points(center_point_guess,size=base_point_size,face_color="grey",border_color="white",blending="additive",name="center_point_guess")
        # viewer.add_points(new_center_guess,size=base_point_size,face_color="yellow",border_color="green",blending="additive",name="new_center_guess")
        # viewer.add_points(ellipsoid_points,size=base_point_size//2,face_color="orange",border_color="red",blending="additive")
        # viewer.add_shapes([spherical_axis_line_data],shape_type='line',edge_color='yellow',edge_width=10,name='sphere_axis',blending='additive')
        # viewer.add_points(closest_point_on_sphere_axis,size=base_point_size,face_color="yellow",border_color="green",blending="additive")
        # viewer.add_shapes([visual_axis_line_data],shape_type='line',edge_color='purple',edge_width=10,name='visual_axis',blending='additive')
        # viewer.add_points(cone_points,size=base_point_size//2,face_color="cyan",border_color="blue",blending="additive")
        # viewer.add_shapes(cone_line_segments,shape_type='line',edge_width=2,edge_color="blue",blending="additive")

        # calculate new imaging motor position for bootleg gradient descent
        delta = learning_rate*initial_imaging_motor_position*dir*magnitude
        print(f"delta: {delta}\n")
        imaging_motor_position = delta+imaging_motor_position
        print(f"new imaging motor position: {imaging_motor_position}\n")

        reference_arm_shift = imaging_motor_position - reference_motor_position # doing final position - initial position allows you to add this component
        reference_arm_shift = (reference_arm_shift * 0.5) / refractive_index
        print(f"new reference arm shift: {reference_arm_shift}\n")

        padding = base_padding + reference_arm_shift
        print(f"padding: {padding}\n")
        if padding < 0:
            dir = 1.
            magnitude = decay * magnitude
            delta = learning_rate*initial_imaging_motor_position*dir*magnitude
            print(f"delta: {delta}\n")
            imaging_motor_position = delta+imaging_motor_position
            print(f"new imaging motor position: {imaging_motor_position}\n")

            reference_arm_shift = imaging_motor_position - reference_motor_position # doing final position - initial position allows you to add this component
            reference_arm_shift = (reference_arm_shift * 0.5) / refractive_index
            print(f"new reference arm shift: {reference_arm_shift}\n")

            padding = base_padding + reference_arm_shift
            


    # generate points for ellipsoid centered on the center of the fit sphere with radius based on foveal distance
    ellipsoid_points = generate_noisy_ellipsoid_sample_data(tuple(min_center_pt),(1.0,1.0,1.0),radius=min_radius,add_noise=False)
    max_ellipsoid_points = generate_noisy_ellipsoid_sample_data(tuple(min_center_pt),(1.0,1.0,1.0),radius=min_max_radius,add_noise=False)

    # calculate line along predicted optical axis
    offset = np.array([0.,min_radius,0.])
    spherical_axis_line_data = [min_center_pt-offset,min_center_pt+offset]

    # calulate vectors for closest point along fit shpere axis to pivot point on pupil plane
    sphere_axis_vector = (min_center_pt+offset) - (min_center_pt-offset)
    axis_start_to_pivot_point_vector = pivot_point_vector - (min_center_pt-offset)
    scalar_projection = np.dot(axis_start_to_pivot_point_vector,sphere_axis_vector) / np.dot(sphere_axis_vector,sphere_axis_vector)
    closest_point_on_sphere_axis = (min_center_pt-offset) + scalar_projection * sphere_axis_vector

    # calculate axial length and anterior chamber depth
    axial_length = np.sqrt((sphere_axis_vector**2).sum()) * pixel_spacing
    anterior_chamber_depth =  np.sqrt(((closest_point_on_sphere_axis - (min_center_pt-offset))**2).sum()) * pixel_spacing
    #print(f"axial length: {axial_length}\nanterior chamber depth: {anterior_chamber_depth}\n")

    # calculate visual axis segment
    visual_axis_line_data = [closest_point_on_sphere_axis,min_curv_fovea.squeeze()]
    #print(spherical_axis_line_data,visual_axis_line_data)

    # generate cone points for scan angle
    cone_points,cone_base_points = create_cone_points((pivot_point_vector).squeeze(),height=min_radius,angle_degrees=scan_angle,num_segments=200)
    #print(f"cone_points: {cone_points}")

    # distance from anterior sphere to pupil plane
    anterior_sphere_to_pupil_plane = np.sqrt(np.sum((closest_point_on_sphere_axis-(min_center_pt-offset))**2))
    print(f"anterior sphere to pupil plane distance: {anterior_sphere_to_pupil_plane} pixels, {anterior_sphere_to_pupil_plane*pixel_spacing}\n")

    cone_line_segments = []
    for base_point in cone_base_points:
        line_segment = np.array([pivot_point_vector.squeeze(),base_point])
        cone_line_segments.append(line_segment)
    
    base_point_size = 80

    viewer.add_points(rpe_points,face_color="white",border_color="gray",size=4)
    viewer.add_points(curv_ret_points,face_color="red",border_color="gray",size=4)
    viewer.add_points(min_curv_rpe,face_color="white",border_color="gray",size=4)
    viewer.add_points(min_min_distort_curv_rpe,face_color="gray",border_color="white",size=4)
    viewer.add_image(thickness_map)

    viewer.add_image(height_map)
    viewer.add_points(pivot_point_vector,size=base_point_size,face_color="green",border_color="yellow",blending="additive",name="pivot_point")
    viewer.add_points(min_ppp_vector,size=base_point_size,face_color="red",border_color="purple",blending="additive",name="projected_pivot_point")
    #viewer.add_points(max_point,size=base_point_size,face_color="purple",border_color="yellow",blending="additive",name="max_point")
    #viewer.add_points(fovea_point,size=40,face_color="purple",border_color="yellow",blending="additive",name="fovea_point")
    viewer.add_points(min_curv_fovea,size=base_point_size,face_color="yellow",border_color="purple",blending="additive",name="curv_fovea_point")
    #viewer.add_points(center_point_guess,size=base_point_size,face_color="grey",border_color="white",blending="additive",name="center_point_guess")
    viewer.add_points(min_center_pt,size=base_point_size,face_color="yellow",border_color="green",blending="additive",name="new_center_guess")
    viewer.add_points(ellipsoid_points,size=base_point_size//2,face_color="orange",border_color="red",blending="additive")
    viewer.add_points(max_ellipsoid_points,size=base_point_size//2,face_color="red",border_color="orange",blending="additive")
    viewer.add_shapes([spherical_axis_line_data],shape_type='line',edge_color='yellow',edge_width=10,name='sphere_axis',blending='additive')
    viewer.add_points(closest_point_on_sphere_axis,size=base_point_size,face_color="yellow",border_color="green",blending="additive")
    viewer.add_shapes([visual_axis_line_data],shape_type='line',edge_color='purple',edge_width=10,name='visual_axis',blending='additive')
    viewer.add_points(cone_points,size=base_point_size//2,face_color="cyan",border_color="blue",blending="additive")
    viewer.add_shapes(cone_line_segments,shape_type='line',edge_width=2,edge_color="blue",blending="additive")

    print(
        f"min std: {min_std}, min radius: {min_radius} pixels, {min_radius*pixel_spacing} mm, min center: {min_center_pt}, min motor position: {min_motor_position}\n"
    )


    '''
    #############################

    (
        rpe_points2, _, curv_rpe_points2, thickness_map2) = generate_sphere_fit(
        label_data=retina_data,
        imaging_motor_position=initial_imaging_motor_position,
        pivot_point=pivot_point,
        scan_angle=scan_angle,
        viewer=viewer,
    )

    ellipsoid_points = generate_noisy_ellipsoid_sample_data(tuple(np.array([142.66984777,1387.6268369,-42.84939724])),(1.0,1.0,1.0),radius=min_radius,add_noise=False)
    #############################
    '''

    viewer.show()
    napari.run()



# class FileGeneratorWidget(Container):
#     """A magicgui widget to generate and step through files in a directory."""

#     def __init__(self):
#         super().__init__()

#         # Widgets for user input
#         self.directory_input = FileEdit(
#             label="Choose a directory",
#             mode='d',  # 'd' specifies directory selection
#             #value=Path.home()
#             value=Path(r"E:\38 peak stage ret_chor crop")
#         )
#         self.extension_input = LineEdit(
#             label="File extension",
#             value="*.npz"
#         )
#         self.generate_button = PushButton(
#             text="Create File Generator"
#         )

#         # Widget to display the current file
#         self.next_file_button = PushButton(
#             text="Next File"
#         )
#         #self.current_file_label = Label(value="No file selected.")

#         self.current_shape:tuple = ()

#         # Layout the widgets
#         self.extend([
#             self.directory_input,
#             self.extension_input,
#             self.generate_button,
#             self.next_file_button,
#             #self.current_file_label
#         ])

#         # State for the file generator
#         self.file_generator = None

#         # Connect button signals to methods
#         self.generate_button.clicked.connect(self._create_generator)
#         self.next_file_button.clicked.connect(self._next_file)

#     def _create_generator(self):
#         """Creates a generator for files with the specified extension."""
#         directory = self.directory_input.value
#         extension = self.extension_input.value

#         if not directory or not Path(directory).is_dir():
#             show_info("Error: Invalid directory.")
#             return

#         # Create a generator for the files
#         self.file_generator = (
#             f for f in Path(directory).rglob(extension)
#             if f.is_file()
#         )
#         show_info(f"Generator created for '{extension}' files in {directory}.")

#     def _next_file(self):
#         """Advances the generator and displays the next file."""
#         if self.file_generator is None:
#             show_info("Please create a generator first.")
#             return

#         try:
#             next_file = next(self.file_generator)
#             show_info(str(next_file))
#             if next_file.suffix == ".npz":
#                 label_data,metadata,layer_type = load_bits_labels_v2(next_file)
#             elif next_file.suffix == ".npy":
#                 label_data = np.load(next_file)
#             show_info(f"{label_data.shape}")
#             #viewer.add_labels(label_data) #,name=metadata["name"],properties=metadata["properties"])

#         except StopIteration:
#             show_info("End of files.")
#             self.file_generator = None  # Reset the generator

# if __name__ == "__main__":
#     # Create the widget and show it
#     viewer = napari.Viewer()
#     my_widget = FileGeneratorWidget()
#     viewer.window.add_dock_widget(my_widget)
#     viewer.show()
#     napari.run()
#     #my_widget.show(run=True)
