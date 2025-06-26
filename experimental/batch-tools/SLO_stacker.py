import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal, Tuple

import napari
import numpy as np
import torch
from magicgui import magicgui

# from tqdm import tqdm
from magicgui.tqdm import tqdm
from napari.qt import thread_worker
from napari_cool_tools_img_proc._equalization_funcs import clahe_pt_func
from napari_cool_tools_img_proc._luminance_funcs import adjust_log_pt_func
from napari_cool_tools_img_proc._normalization_funcs import (
    normalize_data_in_range_pt_func,
    standardize_data_func,
)
from scipy.ndimage import fourier_shift
from torchvision.transforms import InterpolationMode, v2


@thread_worker(progress=True)  # give us an indeterminate progress bar
def ndarray_tofile_thread(path, data):
    """Thread wrapper around numpy.save() function
    Args:
        path(str or list of str): Path to file, or list of paths
        data(ndarray): Data from napari layer to be saved
    Returns:
        None saves ndarray data to .npy file designated by path
    """
    print(f".prof save thread has started for\n{path}.")
    print(f"data min/max: {data.min()}/{data.max()}\ndtype: {data.dtype}\n")
    data.tofile(path)
    # np.save(path,data)
    print(f".prof save thread has completed for\n{path}.")
    return


@magicgui(
    SLO_dir={"label": "SLO Directory", "mode": "d"},
    output_dir={"label": "Output Directory", "mode": "d"},
    call_button="SLO Stacker",
)
def generate_fast_curve_correction(
    SLO_dir: Path = Path(r"D:\JJ\Projects\SLO_Data\2025\\"),
    output_dir: Path = Path("D:\JJ\Projects\SLO_Data\Stacked\\"),
    output_file_suffix: str = "stacked",
    align_ascans: bool = True,
    subpixel_reg: bool = True,
    shift_direction: Literal[-1, 1] = -1,
    scale_factor: Tuple[float, float] = (1.0, 0.25),
    stack: bool = True,
    process_SLO: bool = True,
    standardize: bool = True,
    normalize: bool = True,
    min_val: float = 0.0,
    max_val: float = 1.0,
    log_adjust: bool = True,
    gain: float = 1.0,
    CLAHE: bool = True,
    clip_limit: float = 2.5,
    save_file: bool = True,
    verbose: bool = False,
):
    """
    Stacks .SLO files along new z axis.

    Parameters
    ----------
    SLO_dir: Directory Containing .SLO files to be converted
    output_dir: Directory to Store processed .SLO files if left empty will save to SLO_dir
    output_file_suffix: suffix to put between orginal file name and .SLO extension

    """
    viewer = napari.Viewer(show=False)
    # Get SLO paths
    slo_paths = list(SLO_dir.rglob("*.SLO"))
    if str(output_dir) == ".":
        output_dir = SLO_dir  # Path.cwd()

    os.makedirs(str(output_dir), exist_ok=True)

    if verbose:
        print(output_dir, list(slo_paths))

    #test_paths = [slo_paths[0]]
    # Iterate through paths
    # for slo_path in test_paths:
    for slo_path in tqdm(slo_paths, desc="Stacking .SLO data"):
        # get metadata
        path_parent = slo_path.parent
        path_stem = slo_path.stem
        xml_path = path_parent / f"{path_stem}.xml"

        # load and correct files
        if verbose:
            print(f"getting meta data from {xml_path}.")
        slo_tree = ET.parse(xml_path)
        root = slo_tree.getroot()
        xml_image_size = root[0].find("Image_Size")
        height = int(xml_image_size.get("Width"))
        width = int(xml_image_size.get("Height"))
        if verbose:
            print(f"WxH: {width}x{height}")

        # define chunks as little endian f32 4 byte floats with HEIGHT values
        # per row and WIDTH values per column
        mip_z = np.dtype(
            ("<f8", (height, width))
        )  # saved as double precision f64 8 byte
        enface = np.fromfile(slo_path, dtype=mip_z, count=-1)

        # orient A-scans to align
        if align_ascans is True:
            display = np.empty_like(enface)
            display[:, ::2, :] = enface[:, ::2, :]
            display[:, 1::2, :] = np.flip(enface[:, 1::2, :], 2)

            # convert from sinusoidal space to linear space
            Xn = np.arange(width)
            x_org = (width / 2) * np.sin(2 * np.pi / (2 * width) * Xn - np.pi / 2) + (
                width / 2
            )

            interp_sin_lin = np.empty_like(enface)

            if verbose:
                print(f"Xn: {Xn}\nx_org:{x_org}\n")

            depth = enface.shape[0]

            for i in range(depth):
                for j in range(height):
                    interp_sin_lin[i, j, :] = np.interp(Xn, x_org, display[i, j, :])

            display[:] = interp_sin_lin[:]

        else:
            display = enface

        if subpixel_reg is True:
            from skimage.registration import phase_cross_correlation

            even = display[:, ::2, :]
            odd = display[:, 1::2, :]

            shift, error, diffphase = phase_cross_correlation(
                even, odd, upsample_factor=100
            )

            input_ = np.fft.fft2(odd)
            result = fourier_shift(
                input_, (0.0, 0.0, shift_direction * shift[2]), axis=2
            )
            result = np.fft.ifft2(result)
            odd_shift = result.real

            if verbose:
                print(f"shift: {shift}\n")

            registered = np.empty_like(display)
            registered[:, ::2, :] = even
            registered[:, 1::2, :] = odd_shift

            display = registered

        # rescale image
        if scale_factor[0] != 1.0 or scale_factor[1] != 1.0:
            scale_factor_t = torch.Tensor(scale_factor)
            display_shape_t = torch.Tensor(display.shape[1:])
            new_shape_t = (scale_factor_t * display_shape_t).to(torch.uint32)
            # new_shape = new_shape_t.round().to(torch.uint32).numpy().astype(np.uint32)
            if verbose:
                print(
                    f"{scale_factor_t} x {display_shape_t} = {new_shape_t}"
                )  #: {new_shape}")
            # new_size = torch.Tensor(scale_factor)*torch.Tensor(display.shape[1:]).round().numpy().astype(np.uint8)
            # print(f"new size: {new_size}")
            display = v2.functional.resize(
                torch.from_numpy(display), new_shape_t, InterpolationMode.BILINEAR
            ).numpy()

        # stack files
        if stack:
            _, scaled_height, scaled_width = display.shape
            if verbose:
                print(f"shape before: {display.shape}")
            display = display.reshape(-1, scaled_height // 2, scaled_width)
            if verbose:
                print(f"shape after: {display.shape}")

        if process_SLO:
            if standardize:
                display = standardize_data_func(display)
            if normalize:
                display = normalize_data_in_range_pt_func(
                    display, min_val=min_val, max_val=max_val
                )
            if log_adjust:
                display = adjust_log_pt_func(display, gain=gain)
            if CLAHE:
                display = clahe_pt_func(display, clip_limit=clip_limit)

        # save files
        viewer.add_image(display, name=slo_path.name)

        if save_file:
            ouput_name = f"{path_stem}_{output_file_suffix}.SLO"
            output_meta_data_name = f"{path_stem}_{output_file_suffix}.xml"
            output_slo_file = output_dir / ouput_name
            output_slo_metadata = output_dir / output_meta_data_name
            if verbose:
                print(
                    f"Saving file:\n{output_slo_file}\nand its metadata:\n{output_slo_metadata}\n"
                )

            # modify metadata and save
            xml_image_size.set("Height", str(display.shape[-2]))
            xml_image_size.set("Width", str(display.shape[-1]))
            slo_tree.write(output_slo_metadata)
            # copyfile(xml_path,output_slo_metadata)

            # save .SLO using threads
            # reverse flip and transpose that occured upon loading
            worker = ndarray_tofile_thread(output_slo_file, display)
            worker.start()

    # viewer.add_labels(cart,name="uncle_ben-s_curve_correction")
    viewer.show()
    napari.run()


generate_fast_curve_correction.show(run=True)
