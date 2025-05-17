"""
This module contains code for measuring volumetric data.
"""

import numpy as np
from math import sqrt
from napari.qt.threading import thread_worker
from napari.layers import Image, Layer
from napari_cool_tools_io import viewer
from magicgui import magicgui

def calc_label_volumes(layer:Layer):
    ''''''
    retina_mask = layer.data == 1
    choroid_mask = layer.data == 2

    retina_voxels = np.count_nonzero(retina_mask)
    choroid_voxels = np.count_nonzero(choroid_mask)

    print(f"retina volume: {retina_voxels} voxels\nchoroid volume: {choroid_voxels} voxels")

def calc_radius(final,init)->int: 
    ''''''
    x = abs(final[0] - init[0])
    y = abs(final[2] - init[2])
    r = sqrt(x*x + y*y)
    return int(r)

def clear_labels(layer):
    ''''''
    layer.data = np.zeros(layer.data.shape,dtype=np.int8)

def draw_circle(layer, x, y, r, label_val=1):
    ''''''
    labels = layer.data
    y, x = np.ogrid[-x:labels.shape[0]-x, -y:labels.shape[1]-y]
    mask = x*x + y*y <= r*r
    labels[mask] = label_val

@magicgui(call_button='Activate')
def project_mask(mask_layer:Layer,labels_layer:Layer):
    ''''''
    i0 = labels_layer.data.shape.index(mask_layer.data.shape[0])
    i1 = labels_layer.data.shape.index(mask_layer.data.shape[1])
    #i0 = mask_layer.data.shape.index(labels_layer.data.shape[0]) + 1
    #i1 = mask_layer.data.shape.index(labels_layer.data.shape[2]) + 1
    print(f"i0: {i0}, i1: {i1}")
    dims = range(labels_layer.data.ndim - 1)
    print(f"dims: {dims}, ndim: {labels_layer.data.ndim}")
    new_dim = list(set((i0,i1)) ^ set(range(labels_layer.data.ndim)))[0]
    new_dim_val = labels_layer.data.shape[new_dim]
    #orig_dims = (i1,new_dim,i0)
    #trans_dim = (new_dim,i0,i1)

    #new_dims = (i0,new_dim,i1)
    #new_dims = (2,0,1)

    mask_3d = np.repeat(mask_layer.data[np.newaxis,:],new_dim_val,axis=0)
    print(f"mask 3d shape: {mask_3d.shape}")

    d0 = mask_3d.data.shape.index(labels_layer.data.shape[0])
    d1 = mask_3d.data.shape.index(labels_layer.data.shape[1])
    d2 = mask_3d.data.shape.index(labels_layer.data.shape[2])

    print(f"match indicies: {d0,d1,d2}, match vals {mask_3d.data.shape[d0],mask_3d.data.shape[d1],mask_3d.data.shape[d2]}")
    new_dims = (d0,d1,d2)#(2,0,1)

    out = np.transpose(mask_3d, new_dims)
    print(f"labels shape: {labels_layer.data.shape}, output shape: {out.shape}")
    result = out * labels_layer.data
    viewer.add_labels(result)


def click_drag(layer, event):
    init_pos = event.position
    clear_labels(layer)

    #layer.selected_label = 1
    print('mouse down')
    print(f'init position: {init_pos}\n')
    dragged = False
    yield
    # on move
    while event.type == 'mouse_move':
        #radius = calc_radius(event.position,init_pos)
        #print('Clear Labels')
        #print(f'Draw circle of radius: {radius} centered at: ({int(init_pos[0])},{int(init_pos[1])})')
        #layer.data[int(event.position[0]),int(event.position[1])] = 1
        #layer.refresh()
        #print(event.position)
        final_pos = event.position
        clear_labels(layer)
        radius = calc_radius(final_pos,init_pos)
        print(f'Draw circle of radius: {radius} centered at: ({int(init_pos[2])},{int(init_pos[0])})')
        draw_circle(layer,init_pos[2],init_pos[0],radius,label_val=1)
        layer.refresh()
        dragged = True
        yield
    # on release
    if dragged:
        #final_pos = event.position
        #radius = calc_radius(final_pos,init_pos)
        #print(f'Draw circle of radius: {radius} centered at: ({int(init_pos[2])},{int(init_pos[0])})')
        #draw_circle(layer,init_pos[2],init_pos[0],radius,label_val=2)
        #layer.refresh()
        viewer.window.add_dock_widget(project_mask,name="projection_mask",area="right")
        layer.mouse_drag_callbacks.remove(click_drag)
        #project_mask.show(run=True)
        print('drag end')
    else:
        layer.mouse_drag_callbacks.remove(click_drag)
        print('clicked!')

def draw_circle_mask(enface:Layer, vol:Layer):
    """Generate maximum intensity projections (MIP) along selected orthoganal image planes from structural OCT data.
    
    Args:
        img (Image): 3D ndarray representing structural OCT data
        xy (bool): Toggle xy plane MIP (enface plane by default)
        yz (bool): Toggle yz plane MIP
        zx (bool): Toggle zx plane MIP
    
    Returns:
        List of napari Layers containing selected MIP planes
    """

    mask_name = f"{enface.name}_mask"
    
    labels_layer = viewer.add_labels(np.zeros(enface.data.shape,dtype=np.int8), name=mask_name)
    labels_layer.mode = 'paint'
    labels_layer.selected_label = 0

    target_shape = enface.data.shape
    #current_shape = vol.data.shape
    order_1 = vol.data.shape.index(target_shape[0])
    order_2 = vol.data.shape.index(target_shape[1])
    order_0 = list(set((order_2,order_1)) ^ set(viewer.dims.order))[0]
    viewer.dims.order = (order_0,order_1,order_2)

    labels_layer.mouse_drag_callbacks.append(click_drag)

    
    return

def mark_fovea(layer, fovea_pos):
    """"""

    print(layer.data.shape)
    layer.data = layer.data[:,0,:]
    print(layer.data.shape)
    draw_circle(layer,fovea_pos[2],fovea_pos[0],10)
    layer.data[int(fovea_pos[2]),int(fovea_pos[0])] = 6
    layer.refresh()

    layer.mouse_drag_callbacks.remove(click_fovea)
    layer.mouse_drag_callbacks.append(click_disc)



def mark_disc(layer, disc_pos):
    """"""

    print(layer.data.shape)
    draw_circle(layer,disc_pos[1],disc_pos[2],10,2)
    layer.data[int(disc_pos[1]),int(disc_pos[2])] = 10
    layer.refresh()
    layer.mouse_drag_callbacks.remove(click_disc)

    draw_z1(layer,10,6)

def draw_z1(layer,disc_int,fovea_int):
    """"""
    disc_mask = layer.data == disc_int
    fovea_mask = layer.data == fovea_int
    disc = disc_mask.nonzero()
    fovea = fovea_mask.nonzero()
    fy,fx = fovea[0][0],fovea[1][0]
    dy,dx = disc[0][0], disc[1][0]

    r = sqrt((dy-fy)*(dy-fy) + (dx-fx)*(dx-fx))
    #For two points on a 2D coordinate plane: d = √ [ (x₂ - x₁)² + (y₂ - y₁)²]

    print(f'disc: {dx,dy}, fovea: {fx,fy}\n')
    draw_circle(layer,dy,dx,2*r)
    layer.refresh()

    viewer.window.add_dock_widget(project_mask,name="projection_mask",area="right")


def click_fovea(layer, event):
    init_pos = event.position
    clear_labels(layer)

    print('mouse down')
    print(f'init position: {init_pos}\n')
    dragged = False
    yield
    # on move
    while event.type == 'mouse_move':

        final_pos = event.position
        #clear_labels(layer)
        #layer.refresh()
        dragged = True
        yield
    # on release
    if dragged:
        final_pos = event.position
        print(f'final position: {final_pos}\n')
        print('drag end')
        mark_fovea(layer,final_pos)
    else:
        print(f'init position: {init_pos}\n')
        print('clicked!')
        mark_fovea(layer,init_pos)

def click_disc(layer, event):
    init_pos = event.position
    #clear_labels(layer)

    print('mouse down')
    print(f'init position: {init_pos}\n')
    dragged = False
    yield
    # on move
    while event.type == 'mouse_move':

        final_pos = event.position
        #clear_labels(layer)
        layer.refresh()
        dragged = True
        yield
    # on release
    if dragged:
        final_pos = event.position
        print(f'final position: {final_pos}\n')
        print('drag end')
        mark_disc(layer,final_pos)
    else:
        print(f'init position: {init_pos}\n')
        print('clicked!')
        mark_disc(layer,init_pos)

def calc_zone1(enface:Layer, vol:Layer):
    """Generate maximum intensity projections (MIP) along selected orthoganal image planes from structural OCT data.
    
    Args:
        img (Image): 3D ndarray representing structural OCT data
        xy (bool): Toggle xy plane MIP (enface plane by default)
        yz (bool): Toggle yz plane MIP
        zx (bool): Toggle zx plane MIP
    
    Returns:
        List of napari Layers containing selected MIP planes
    """

    mask_name = f"{enface.name}_mask"
    
    labels_layer = viewer.add_labels(np.zeros(vol.data.shape,dtype=np.int8), name=mask_name)
    labels_layer.mode = 'paint'
    labels_layer.selected_label = 0

    """
    target_shape = enface.data.shape
    #current_shape = vol.data.shape
    order_1 = vol.data.shape.index(target_shape[0])
    order_2 = vol.data.shape.index(target_shape[1])
    order_0 = list(set((order_2,order_1)) ^ set(viewer.dims.order))[0]
    viewer.dims.order = (order_0,order_1,order_2)
    """

    labels_layer.mouse_drag_callbacks.append(click_fovea)
    
    return