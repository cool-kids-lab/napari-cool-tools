"""
This module contains code for calculating and manipulating projections of volumetric data.
"""
from typing import List
from napari.utils.notifications import show_info
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari_cool_tools_io import viewer

def mip(img:Image,yx=True,zy=False,xz=False):
    """Generate maximum intensity projections (MIP) along selected orthoganal image planes from structural OCT data.
    
    Args:
        img (Image): 3D ndarray representing structural OCT data
        xy (bool): Toggle xy plane MIP (enface plane by default)
        yz (bool): Toggle yz plane MIP
        zx (bool): Toggle zx plane MIP
    
    Returns:
        List of napari Layers containing selected MIP planes
    """

    worker = mip_thread(img=img,yx=yx,zy=zy,xz=xz)
    
    return

@thread_worker(connect={"yielded": viewer.add_layer})
def mip_thread(img:Image,yx=True,zy=False,xz=False) -> List[Layer]:
    """Generate maximum intensity projections (MIP) along selected orthoganal image planes.
    
    Args:
        img (Image): 3D ndarray to calulate maximum intensity projection from
        xy (bool): Toggle xy plane MIP (enface plane by default)
        yz (bool): Toggle yz plane MIP
        zx (bool): Toggle zx plane MIP
    
    Yields:
        List of napari Layers containing selected MIP planes
    """

    show_info(f'Maximum Intensity Projection thread has started')
    data = img.data
    name = img.name
    layer_type = "image"

    if yx == True:
        mip_yx_name = f"MIP_xy_{name}"
        yx = data.transpose(1,2,0)
        mip_yx = yx.max(0)
        add_kwargs = {"name": f"{mip_yx_name}"}
        layer = Layer.create(mip_yx,add_kwargs,layer_type)
        yield layer
    if zy == True:
        mip_zy_name = f"MIP_yz_{name}"
        mip_zy = data.max(0)
        add_kwargs = {"name": f"{mip_zy_name}"}
        layer = Layer.create(mip_zy,add_kwargs,layer_type)
        yield layer
    if xz == True:
        mip_xz_name = f"MIP_xz_{name}"
        xz = data.transpose(1,0,2)
        mip_xz = xz.max(2)
        add_kwargs = {"name": f"{mip_xz_name}"}
        layer = Layer.create(mip_xz,add_kwargs,layer_type)
        yield layer

    show_info(f'Maximum Intensity Projection thread has completed')