""""""

import numpy as np
import pyvista as pv
from napari_cool_tools_img_proc._normalization_funcs import normalize_data_in_range_func
from numpy import pi

#from jj_nn_framework.image_funcs import normalize_in_range

# math constants
delta_rad = pi/3.0 
e = np.exp(1)
x0,y0,z0 = 0,0,0

res = 17

# make 3D grid
x = np.arange(-(res-1),res-1)
y = np.arange(-(res-1),res-1)

coords = np.ogrid[0:res,0:res,0:res]
x,y,z = [(coord-4) for coord in coords]
vol_mask = (x**2+y**2+z**2) <= 3

epsilon = 10.0**(-6)
density = x**2.0+y**2.0+z**2.0

#density = (1 / ((x**2.0+y**2.0+z**2.0)+epsilon))

density = normalize_data_in_range_func(density,min_val=0.0,max_val=1.0)

#density = (vol_mask*1.0).astype(np.float32) * density

vol = pv.ImageData(dimensions=density.shape)
vol.point_data["density"] = density.flatten()

v_plot = pv.Plotter()
v_plot.add_volume(vol,cmap="fire",scalars="density",opacity='linear')
v_plot.show()

#end_position = (1**2+1**2)**(1/2)
#norm_end = np.linalg.norm(np.array([0.5,0.5]))
#line = pv.Line((0,0,0), (-norm_end,norm_end,0), resolution=7)
#line.point_data["position"] = range(8)
#line2 = pv.Line((0,0,0), (norm_end,norm_end,0), resolution=7)
#line2.point_data["position"] = range(8)
#
#x_axis = pv.Line((0,0,0),(1,0,0),resolution=1)
#y_axis = pv.Line((0,0,0),(0,1,0))
#
#plotter = pv.Plotter()
#plotter.add_mesh(line,name="orginal",cmap="fire",line_width=2)
#plotter.add_mesh(line2,cmap="CET_L16",line_width=2)
#plotter.add_mesh(x_axis,color="red")
#plotter.add_mesh(y_axis,color="cyan")
##line.plot(cmap="fire",line_width=10)
#plotter.show()