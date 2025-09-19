"""
"""

from dataclasses import dataclass

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
#import sympy as sp
import napari

@dataclass
class OCT_Settings():
    slow_axis:int=840
    axial_axis:int=2016
    fast_axis:int=800
    imaging_range:float=12.
    scan_angle:float=100.
    refractive_index:float=1.33

def generate_noisy_ellipse_sample_data(center:tuple[int,int]=(0,0),semi_axes:tuple[float,float,float]=(1.0,1.0),radius:float=1.0,theta_samples:int=20,seed:int=42,add_noise:bool=True):
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
    #v = np.linspace(0.,np.pi, theta_samples)
    #u,v = np.meshgrid(u,v,sparse=True)
    a = semi_axes[0] #1.0
    b = semi_axes[1] #0.5,
    #c = semi_axes[2] #800/840

    # calculate cartesian coordinates from polar
    x = a*np.cos(u)*radius
    y = b*np.sin(u)*radius

    x = x.flatten() + noise
    y = y.flatten() + noise

    x = x+center[0]
    y = y+center[1]

    return np.column_stack((x,y))

if __name__ == "__main__":

    label_data = np.zeros((840,2016))
    print(label_data.shape)

    r = 7.5+(1/7)
    center = r,0.0

    oct_settings = OCT_Settings(scan_angle=100)
    imaging_range = oct_settings.imaging_range
    refractive_index = oct_settings.refractive_index
    scan_angle = oct_settings.scan_angle
    slow_axis = oct_settings.slow_axis
    axial_axis = oct_settings.axial_axis
    fast_axis = oct_settings.fast_axis

    # map memory index to appropriate scan angle - assuming linear relationship
    fast_axis_angle_mapping = np.linspace(-(scan_angle/2.),scan_angle/2.0,fast_axis)
    slow_axis_angle_mapping = np.linspace(-(scan_angle/2.),scan_angle/2.0,slow_axis)

    # normalize indicies between -100 and 100
    # fast_axis_norm_100 = np.linspace(-100,100,fast_axis)
    # slow_axis_norm_100 = np.linspace(-100,100,slow_axis)
    fast_axis_norm = np.linspace(-1.,1.,fast_axis)
    slow_axis_norm = np.linspace(-1.,1.,slow_axis)

    for idx in range(fast_axis):
        #print(f"{fast_axis_angle_mapping[idx]} deg ({fast_axis_angle_mapping[idx]*(2/360)})pi radians angle at index {idx}, normalized({fast_axis_norm_100[idx]})\n")
        print(f"{fast_axis_angle_mapping[idx]} deg ({fast_axis_angle_mapping[idx]*(2/360)})pi radians angle at index {idx}, normalized({fast_axis_norm[idx]})\n")

    # apply Yakub's scan angle fitting correction
    def scan_angle_fit_func(x,bb:float=0.7669,cc:float=0.05,dd=0.0063,ee=0.0107):
        """"""
        sign = np.sign
        #return bb*sign(x)*abs(x/100.)**1+cc*sign(x)*abs(x/100.)**2+dd*sign(x)*abs(x/100.)**3+ee*sign(x)*abs(x/100.)**4
        return bb*sign(x)*abs(x)**1+cc*sign(x)*abs(x)**2+dd*sign(x)*abs(x)**3+ee*sign(x)*abs(x)**4
    
    coefficient_values = [0.7669,0.05,0.0063,0.0107]
    coeff_lower_bounds = [0.7129,-0.2076,-0.3778,-0.1715]
    coeff_upper_bounds = [0.8209,0.3075, 0.3914,0.1928]
    parameter_bounds = (coeff_lower_bounds,coeff_upper_bounds)

    #fast_axis_nonlinear_angle_radian_mapping = scan_angle_fit_func(fast_axis_norm_100,*coefficient_values) # mapping is in radians
    fast_axis_nonlinear_angle_radian_mapping = scan_angle_fit_func(fast_axis_norm,*coefficient_values) # mapping is in radians
    radians_to_degree = 180/np.pi # 360/pi because we use half of the angle per side
    fast_axis_nonlinear_angle_degree_mapping = fast_axis_nonlinear_angle_radian_mapping * radians_to_degree

    mean_diff = fast_axis_angle_mapping - fast_axis_nonlinear_angle_degree_mapping
    mean_std = np.sqrt((mean_diff**2).mean())

    #popt, pcov = curve_fit(scan_angle_fit_func,fast_axis_norm_100,fast_axis_nonlinear_angle_radian_mapping,bounds=parameter_bounds)
    print(mean_diff,mean_std)
        
    import matplotlib.pyplot as plt
    #plt.plot(fast_axis_norm_100,fast_axis_nonlinear_angle_radian_mapping,label='Non-linear scan angle fit')
    plt.plot(fast_axis_norm,fast_axis_nonlinear_angle_radian_mapping,label='Non-linear scan angle fit')
    plt.show()
    import sys
    sys.exit(0)
    ###########

    imaging_range /= refractive_index
    pixel_spacing = imaging_range / axial_axis #  mm/pixel along scanning axis
    #pixel_spacing = imaging_range / label_data.shape[1] #z_shape
    print(f"pixel_spacing: {pixel_spacing}\n")

    pivot_point = np.array([1.2,0.])
    #circle_center = np.array([0.,0.])
    circle_center = np.array(center)
    circle_points = generate_noisy_ellipse_sample_data(center=center,semi_axes=(1.0,1.0),radius=r,theta_samples=200,add_noise=False)
    #print(f"circle points: {circle_points}\n")

    dx_dy = (scan_angle/2.0/90)
    # m = (scan_angle/2.0/90)

    x_init,y_init = pivot_point[0],pivot_point[1]
    dx = 1
    dy = 1 / dx_dy

    dx2 = 0
    dy2 = 1
    # samples = 200
    # x = np.linspace(x_init,x_init+ r*dx,samples)
    # y = np.linspace(y_init,y_init+ r*dy,samples)

    # points = []
    # flip_points = []
    # for idx,val in enumerate(x):
    #     points.append(np.array((val,y[idx])))
    #     flip_points.append(np.array((val,y[idx]*-1)))
    # points = np.stack(points,axis=0)
    # #print(f"points: {points}\n")

    a = dx**2 + dy**2
    b = 2*(dx*(x_init-center[0]) + dy*(y_init-center[1]))
    c = (x_init-center[0])**2 + (y_init-center[1])**2 - r**2
    discriminant = b**2 - 4*a*c

    a2 = dx2**2 + dy2**2
    b2 = 2*(dx2*(x_init-center[0]) + dy2*(y_init-center[1]))
    c2 = (x_init-center[0])**2 + (y_init-center[1])**2 - r**2
    discriminant2 = b2**2 - 4*a2*c2

    pivot_intersection_pts = []
    if discriminant > 0:
        print("I am here now!!")
        t_values = np.roots([a,b,c])
        print(f"roots: {t_values}")
        for t in t_values:
            if t >= 0:
                ix = x_init + t * dx
                iy = y_init + t * dy
                pivot_intersection_pts.append(np.array([ix,iy]))
                pivot_intersection_pts.append(np.array([ix,-iy]))

    print(f"pivot intersection points: {pivot_intersection_pts}\n")

    pupil_intersection_pts = []
    if discriminant2 > 0:
        print("I am here now!!")
        t_values2 = np.roots([a2,b2,c2])
        print(f"roots: {t_values2}")
        for t2 in t_values2:
            if t2 >= 0:
                ix2 = x_init + t2 * dx2
                iy2 = y_init + t2 * dy2
                pupil_intersection_pts.append(np.array([ix2,iy2]))
                pupil_intersection_pts.append(np.array([ix2,-iy2]))

    print(f"pupil intersection points: {pupil_intersection_pts}\n")

    pivot_lines = []
    for intersection in pivot_intersection_pts:
        pivot_lines.append(np.array([intersection,pivot_point]))

    pupil_lines = []
    for intersection in pupil_intersection_pts:
        pupil_lines.append(np.array([intersection,pivot_point]))

    center_lines = []
    for intersection in pivot_intersection_pts:
        center_lines.append(np.array([intersection,circle_center]))

    center_vector_positive = pivot_intersection_pts[0] - circle_center
    center_vector_negative = pivot_intersection_pts[1] - circle_center
    print(f"center_vector_positive: {center_vector_positive}\ncenter_vector_negative: {center_vector_negative}")

    center_vector_dot = np.dot(center_vector_positive,center_vector_negative)
    cvp_magnitude = np.linalg.norm(center_vector_positive)
    cvn_magnitude = np.linalg.norm(center_vector_negative)

    center_cosine_angle = center_vector_dot/(cvp_magnitude*cvn_magnitude)
    center_angle_radians = np.arccos(np.clip(center_cosine_angle, -1.0,1.0))
    center_angle_degrees = np.degrees(center_angle_radians)
    print(f"center dot: {center_vector_dot}")
    print(f"center angle: {center_angle_degrees}")
    print(f"magnitudes: {cvp_magnitude*cvn_magnitude}\n")
    print(f"center cosine angle: {center_cosine_angle}\n")

    # a = (x-center[0])**2 + (y-center[1])**2
    # b = 2*x*(x_init-center[0]) + 2*y*(y_init-center[1])
    # c = (x_init-center[0])**2 + (y_init-center[1])**2 - r**2

    # discriminant = b**2 - (4*a*c)

    # greater_than_zero = discriminant > 0

    # #discriminant = discriminant[greater_than_zero]

    # t = 2*c / (-b + np.sqrt(b**2-(4*a*c)))

    # print(f"x: {x}\ny: {y}\n")
    # print(f"x_init-center[0]: {x_init-center[0]}\n")
    # print(f"y_init-center[0]: {y_init-center[1]}\n")
    # print(f"x_init-center[0]: {(x_init-center[0])**2}\n")
    # print(f"y_init-center[0]: {(y_init-center[1])**2}\n")

    # print(f"a: {a}\n")
    # print(f"b: {b}\n")
    # print(f"c: {c}\n")
    # print(f"discriminant: {discriminant}\n")

    # print(f"t: {t}\n")
    
    # define symbols
    # x,y = sp.symbols('x y')

    # define equations

    # dir_vector = np.array([2*(50/90)*pivot_point[0]+pivot_point[0],2*pivot_point[0]+pivot_point[1]])
    # dir_vector2 = np.array([2*pivot_point[0]+pivot_point[0],2*pivot_point[0]+pivot_point[1]])

    # m = 1. - (scan_angle/2.0/90)
    # m = (scan_angle/2.0/90)
    # print(f"slope: {m}\n")
    # line = sp.Eq(m*x+y, r)
    # #line2 = sp.Eq(m2*x+y, r)
    # circle = sp.Eq((x-r)**2 + (y-0.)**2,r**2)

    # solve system
    # solutions = sp.solve((line,circle),(x,y))
    #solutions2 = sp.solve((line2,circle),(x,y))
    # print(f"solutions: {solutions}\n")
    # angle_segments = []
    # for solution in solutions:
    #     point = np.array(solution)
    #     print(f"solution point: {point}\n")
    #     #if (pivot_point[0] - solution[0]) / (pivot_point[1] - solution[1]) <= m:
    #     angle_segments.append(np.array([pivot_point.squeeze(),point]))    
    #     angle_segments.append(np.array([pivot_point.squeeze(),point*np.array([1.,-1.])]))    


    viewer = napari.Viewer(show=False)
    viewer.add_points(pivot_point,size=0.1,face_color="green",border_color="yellow",blending="additive")
    viewer.add_points(circle_center,size=0.1,face_color="yellow",border_color="green",blending="additive")
    viewer.add_points(circle_points,size=0.1,face_color="orange",border_color="red",blending="additive")
    # viewer.add_points(dir_vector,size=0.1,face_color="magenta",border_color="purple",blending="additive")
    # viewer.add_points(dir_vector2,size=0.1,face_color="purple",border_color="magenta",blending="additive")
    # viewer.add_points(points,size=0.1,face_color="cyan",border_color="blue",blending="additive")
    # viewer.add_points(flip_points,size=0.1,face_color="cyan",border_color="blue",blending="additive")
    viewer.add_points(pivot_intersection_pts,size=0.1,face_color="red",border_color="yellow",blending="additive")
    viewer.add_points(pupil_intersection_pts,size=0.1,face_color="green",border_color="yellow",blending="additive")
    viewer.add_shapes(pivot_lines,shape_type="line",edge_width=0.05,edge_color="red",blending="additive")
    viewer.add_shapes(pupil_lines,shape_type="line",edge_width=0.05,edge_color="green",blending="additive")
    viewer.add_shapes(center_lines,shape_type="line",edge_width=0.05,edge_color="yellow",blending="additive")
    # viewer.add_shapes(angle_segments,shape_type='line',edge_width=0.1,edge_color="blue",blending="additive")

    viewer.show()
    napari.run()