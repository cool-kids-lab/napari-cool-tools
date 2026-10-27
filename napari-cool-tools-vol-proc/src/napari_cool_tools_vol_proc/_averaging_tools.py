"""
This module contains code for averaging 2D slices
"""

from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from napari_cool_tools_io import viewer
from napari_cool_tools_registration.__init__ import RegistrationMode
from napari_cool_tools_oct_preproc.__init__ import OCTACalc

from napari_cool_tools_vol_proc._averaging_tools_funcs import (
    Implementation,
    average_bscans,
    average_per_bscan,
    average_per_bscan_pt,
)


def average_bscans_plugin(vol: Image, scans_per_avg: int = 3):
    """Function averaging every scans_per_avg images/B-scans togehter.
    Args:
        vol (Image): vol representing volumetric or image stack data
        scans_per_avg (int): number of consecutive images/B-scans to average together

    Returns:
        Layer volume where values have been averaged every scans_per_avg images/B-scans along the depth dimension
    """

    average_bscans_thread(vol=vol, scans_per_avg=scans_per_avg)

    return


@thread_worker(connect={"returned": viewer.add_layer})
def average_bscans_thread(vol: Image, scans_per_avg: int = 3) -> Layer:
    """Function averaging every scans_per_avg images/B-scans togehter.
    Args:
        vol (Image): vol representing volumetric or image stack data
        scans_per_avg (int): number of consecutive images/B-scans to average together

    Returns:
        Layer volume where values have been averaged every scans_per_avg images/B-scans along the depth dimension
    """

    name = f"{vol.name}_avg_{scans_per_avg}"
    add_kwargs = {"name": name}
    layer_type = "image"
    show_info("Average B-scans thread has started")
    avg_data = average_bscans(vol.data, scans_per_avg=scans_per_avg)
    show_info("Average B-scans thread has completed")
    layer = Layer.create(avg_data, add_kwargs, layer_type)

    return layer

def average_per_bscan_plugin(
    vol: Image,
    scans_per_avg: int = 3,
    registration: bool = False,
    registration_mode: RegistrationMode = RegistrationMode.Phase_Correlation,
    OCTA: bool = False,
    octa_mode: OCTACalc = OCTACalc.STD,
    kernel_averaging: bool = False,
    slow_window: int = 3, #lateral window size in z direction
    fast_window: int = 3, #lateral window size
    axial_window: int = 3, #axial window size
    sigma_slow: float = 0.0,
    sigma_fast: float = 0.0,
    sigma_axial: float = 0.0,
    enable_axial_filter: bool = True,
):
    """Function averaging every scans_per_avg images/B-scans centered around each image/b-scan.
    Args:
        vol (Image): vol representing volumetric or image stack data
        scans_per_avg (int): number of consecutive images/B-scans to average together
        trim: (bool): Flag indicating that ends should be trimmed if image/B-scan index is less than (scans_per_avg - 1 / 2)

    Returns:
        Layer volume where values at each index each slice is an average of the surrounding bscans from vol
    """

    average_per_bscan_thread(
        vol=vol,
        scans_per_avg=scans_per_avg,
        registration=registration,
        registration_mode=registration_mode,
        OCTA=OCTA,
        octa_mode=octa_mode,
        kernel_averaging=kernel_averaging,
        slow_window=slow_window,
        fast_window=fast_window,
        axial_window=axial_window,
        sigma_slow=sigma_slow,
        sigma_fast=sigma_fast,
        sigma_axial=sigma_axial,
        enable_axial_filter=enable_axial_filter
    )
    

@thread_worker(connect={"yielded": viewer.add_layer})
def average_per_bscan_thread(
    vol: Image,
    scans_per_avg: int = 3,
    registration: bool = False,
    registration_mode: RegistrationMode = RegistrationMode.Phase_Correlation,
    OCTA: bool = False,
    octa_mode: OCTACalc = OCTACalc.STD,
    kernel_averaging: bool = False,
    slow_window: int = 3, #lateral window size in z direction
    fast_window: int = 3, #lateral window size
    axial_window: int = 3, #axial window size
    sigma_slow: float = 0.0,
    sigma_fast: float = 0.0,
    sigma_axial: float = 0.0,
    enable_axial_filter: bool = True,

):
    """Function averaging every scans_per_avg images/B-scans centered around each image/b-scan.
    Args:
        vol (Image): vol representing volumetric or image stack data
        scans_per_avg (int): number of consecutive images/B-scans to average together
        trim: (bool): Flag indicating that ends should be trimmed if image/B-scan index is less than (scans_per_avg - 1 / 2)

    Returns:
        Layer volume where values at each index each slice is an average of the surrounding bscans from vol
    """

    name = f"{vol.name}_{scans_per_avg}_per"
    add_kwargs = {"name": name}
    layer_type = "image"
    show_info("Sliding Window B-scan average thread has started")

    avg_data, octa_data = average_per_bscan_pt(
        vol.data,
        scans_per_avg=scans_per_avg,
        registration=registration,
        registration_mode=registration_mode,
        OCTA=OCTA,
        octa_mode=octa_mode,
        kernel_averaging=kernel_averaging,
        slow_window=slow_window,
        fast_window=fast_window,
        axial_window=axial_window,
        sigma_slow=sigma_slow,
        sigma_fast=sigma_fast,
        sigma_axial=sigma_axial,
        enable_axial_filter=enable_axial_filter
    )

    show_info("Sliding Window B-scan average thread has completed")
    avg_layer = Layer.create(avg_data, add_kwargs, layer_type)
    yield avg_layer

    if octa_data is not None:
        octa_name = f"{vol.name}_{scans_per_avg}_octa"
        octa_add_kwargs = {"name": octa_name}
        octa_layer_type = "image"
        octa_layer = Layer.create(octa_data, octa_add_kwargs, octa_layer_type)
        yield octa_layer

    
