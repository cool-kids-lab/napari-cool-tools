from pathlib import Path
from typing import Tuple

import napari
import numpy as np
import polarTransform
import torch
from magicgui import magicgui
from napari.types import ImageData
from napari_cool_tools_img_proc._equalization_funcs import DTYPE, init_bscan_preproc
from napari_cool_tools_img_proc._normalization_funcs import normalize_data_in_range_func
from skimage.transform import rotate
from torchvision.transforms.v2.functional import InterpolationMode, resize
from tqdm import tqdm
from vispy.color import ColorArray, Colormap


def mask_to_color(img_data: ImageData, mask: ImageData, color: ColorArray):
    """ """
    same_dims = np.ones((img_data.ndim)).astype(np.uint8)
    out_data = np.tile(np.zeros_like(img_data), (3, *same_dims))
    print(f"color: {color}")
    out_data[0][mask] = color.rgb[0, 0]
    out_data[1][mask] = color.rgb[0, 1]
    out_data[2][mask] = color.rgb[0, 2]

    dims = np.arange(out_data.ndim)
    out_data = out_data.transpose(*dims[1:], 0)
    return out_data


def cartify_numpy(
    img_data: ImageData, spline_interpolation_order: int = 3
) -> ImageData:
    # image = io.imread(image_path)
    # print(image.shape)

    out_data = []

    # if img_data.ndim == 2:
    #     img_data = img_data.reshape(-1,*img_data.shape[-2:])
    if img_data.ndim >= 2:
        img_data = img_data.reshape(-1, *img_data.shape[-2:])
        print(f"img_data ndim/shape: {img_data.ndim}/{img_data.shape}")

        for _, b_scan in tqdm(enumerate(img_data), desc="processing slices"):
            # print(f"img_data shape: {b_scan.shape}")
            image = normalize_data_in_range_func(
                b_scan, min_val=0.0, max_val=255.0
            ).astype(np.uint8)
            # image = np.tile(image,(3,1,1)).transpose(1,2,0)
            # print(f"image shape: {image.shape}")

            width, height = image.shape[-2:]
            ####width, height,channel = image.shape

            image = rotate(
                image, 90, resize=True
            )  # to turn theta axis to Y axis as needed for polarTransform
            height_pad = int(
                round(1.66 * height * 2)
            )  # estimates for being in water and typical ref arm position
            ####image = np.pad(image, [(0,0), (height_pad,0), (0,0)] , mode = 'constant', constant_values = 0)
            # image = np.pad(image, [(height_pad,0), (0,0)] , mode = 'constant', constant_values = 0)
            image = np.pad(
                image, [(0, 0), (height_pad, 0)], mode="constant", constant_values=0
            )

            # print(image.shape)

            ####cart_image, ptSettings = polarTransform.convertToCartesianImage(image, initialAngle = -51*np.pi/180, finalAngle = 51*np.pi/180, hasColor = True)
            cart_image, ptSettings = polarTransform.convertToCartesianImage(
                image,
                initialAngle=-51 * np.pi / 180,
                finalAngle=51 * np.pi / 180,
                hasColor=False,
                order=spline_interpolation_order,
            )

            cart_image = rotate(cart_image, -90, resize=True)
            # print(cart_image.shape)

            ####cart_image = cart_image[:,:,:3]

            # print(cart_image.shape)
            cart_image = np.clip(cart_image, 0, 1)

            out_data.append(cart_image)
            # plt.imshow(cart_image)
            # plt.show()
            # plt.imsave("cartesian.png", cart_image)

        output = np.stack(out_data, axis=0).squeeze()

    return output  # cart_image, ptSettings


@magicgui(
    image_path={"label": "Image File", "mode": "r"},
    label_path={"label": "Label File", "mode": "r"},
    output_dir={"label": "Output Directory", "mode": "d"},
    input_HW={"label": "Input HxW", "options": {"min": 0, "max": 10000}},
    processed_HW={"label": "Input HxW", "options": {"min": 0, "max": 10000}},
    retina_mask_range={"label": "Retina Mask Range", "options": {"step": 0.001}},
    choroid_mask_range={"label": "Choroid Mask Range", "options": {"step": 0.001}},
    # processed_HW={"label": "Input HxW","min":0,"max":10000},
    call_button="Uncle Ben's Fast 2D Curve Corrector",
)
def generate_fast_curve_correction(
    image_path: Path = Path(
        r"D:\JJ\Projects\Segmentation_Paper\Figures\Imaging_Data\b-scan_fold_4\imgs.prof"
    ),
    label_path: Path = Path(
        r"D:\JJ\Projects\Segmentation_Paper\Figures\Imaging_Data\b-scan_fold_4\prs.prof"
    ),
    output_dir: Path = Path(
        r"D:\JJ\Projects\Segmentation_Paper\Data\Bscan\Figure_Sample_Scans"
    ),
    output_filename: str = "output.pt",
    input_HW: Tuple[int, int] = [992, 800],
    processed_HW: Tuple[int, int] = [864, 864],
    data_processing_range: slice = slice(212, 213, 1),  # slice(215,218,1),
    preprocess_data: bool = False,  # True,
    process_as_labels: bool = False,
    use_color_map: bool = False,
    retina_color: Tuple[float, float, float] = (0.5, 0.0, 0.0),
    choroid_color: Tuple[float, float, float] = (0.0, 0.5, 0.5),
    retina_mask_range: Tuple[float, float] = (0.497, 0.501),  # (0.497,0.499),
    choroid_mask_range: Tuple[float, float] = (0.501, 1.0),  # (0.499,1.000),
    # sweep:int = 102,
    # downsampling:int = 3,
    # resolution:int = 1,
    # threshold:int = 15, #60
    # isovalue:int = 3,
    # save:bool = False,
    # use_gpu: bool = True
):
    """ """

    # if not process_as_labels:
    #     spline_interploation_order = 3
    # else:
    #     spline_interploation_order = 0

    viewer = napari.Viewer(show=False)
    viewer.open(image_path, plugin="napari-cool-tools-io")
    image_data = viewer.layers[-1].data.copy()
    viewer.open(label_path, plugin="napari-cool-tools-io")
    label_data = viewer.layers[-1].data.copy()
    # normalized_data = normalize_data_in_range_func(image_data,0,255).astype(np.uint8)
    if preprocess_data:
        preproc_data = init_bscan_preproc(
            image_data,
            num_std=16,
            min_intensity=0.0,
            max_intensity=255.0,
            dtype=DTYPE.NP_UINT8,
        )
    else:
        preproc_data = image_data

    init_shape = (image_data.shape[-2], image_data.shape[-1])

    padding_width = init_shape[-1] - input_HW[-1]
    padding_height = init_shape[-2] - input_HW[-2]

    if padding_width > 0:
        pad_per_side = padding_width // 2
        width_start = pad_per_side
        width_end = (init_shape[-1] - 1) - pad_per_side
        preproc_data = preproc_data[:, :, width_start:width_end]
        label_data = label_data[:, :, width_start:width_end]

    if padding_height > 0:
        pad_per_top_bottom = padding_width // 2
        height_start = pad_per_top_bottom
        height_end = (init_shape[-2] - 1) - pad_per_top_bottom
        preproc_data = preproc_data[:, height_start:height_end, :]
        label_data = label_data[:, height_start:height_end, :]

    if preproc_data.dtype == np.float32:
        original_shape_data = resize(
            torch.Tensor(preproc_data),
            input_HW,
            interpolation=InterpolationMode.BILINEAR,
        ).numpy()
        original_label_shape_data = resize(
            torch.Tensor(label_data),
            input_HW,
            interpolation=InterpolationMode.NEAREST_EXACT,
        ).numpy()
    elif preproc_data.dtype == np.uint8:
        original_shape_data = resize(
            torch.Tensor(preproc_data),
            input_HW,
            interpolation=InterpolationMode.BILINEAR,
        ).numpy()
        original_label_shape_data = resize(
            torch.Tensor(label_data),
            input_HW,
            interpolation=InterpolationMode.NEAREST_EXACT,
        ).numpy()
    else:
        raise ValueError(
            f"Image data of dtype {preproc_data.dtype} is not supported, {np.float32},{np.uint8} are supported dtypes."
        )

    # output_data,settings = cartify_numpy(original_shape_data[370])
    # output_data = cartify_numpy(original_shape_data[370])
    output_data = cartify_numpy(
        original_shape_data[data_processing_range], spline_interpolation_order=3
    )
    label_output_data = cartify_numpy(
        original_label_shape_data[data_processing_range], spline_interpolation_order=0
    )

    # og_output_data = original_shape_data[data_processing_range].squeeze()
    og_output_data = normalize_data_in_range_func(
        original_shape_data[data_processing_range], min_val=0.0, max_val=1.0
    )
    og_output_data = og_output_data.squeeze()
    # og_label_output_data = original_label_shape_data[data_processing_range].squeeze()
    og_label_output_data = normalize_data_in_range_func(
        original_label_shape_data[data_processing_range], min_val=0.0, max_val=1.0
    )
    og_label_output_data = og_label_output_data.squeeze()

    print(type(image_data), image_data.shape)
    print(f"Preproc data shape: {image_data.shape}")
    print(
        f"Original data shape: {original_shape_data.shape},{original_shape_data.dtype}"
    )
    print(
        f"Output data shape: {output_data.shape},{output_data.dtype},{output_data.min()},{output_data.mean()},{output_data.max()}"
    )
    print(
        f"Label output data shape: {label_output_data.shape},{label_output_data.dtype},{label_output_data.min()},{label_output_data.mean()},{label_output_data.max()}\n"
    )
    # del viewer
    # gc.collect()
    print(
        "og_output_data",
        type(og_output_data),
        og_output_data.shape,
        og_output_data.dtype,
        og_output_data.min(),
        og_output_data.mean(),
        og_output_data.max(),
    )
    print(
        "og_label_output_data",
        type(og_label_output_data),
        og_label_output_data.shape,
        og_label_output_data.dtype,
        og_label_output_data.min(),
        og_label_output_data.mean(),
        og_label_output_data.max(),
    )

    retina_mask = (label_output_data > retina_mask_range[0]) & (
        label_output_data < retina_mask_range[1]
    )
    # retina_mask = (output_data > 0.47) & (output_data < 0.472)
    choroid_mask = (label_output_data >= choroid_mask_range[0]) & (
        label_output_data <= choroid_mask_range[1]
    )
    # choroid_mask = (output_data >= 0.472) & (output_data <= 1.0)

    color_retina = mask_to_color(
        label_output_data, retina_mask, ColorArray(retina_color)
    )
    color_choroid = mask_to_color(
        label_output_data, choroid_mask, ColorArray(choroid_color)
    )

    og_retina_mask = (og_label_output_data > retina_mask_range[0]) & (
        og_label_output_data < retina_mask_range[1]
    )
    og_choroid_mask = (og_label_output_data > choroid_mask_range[0]) & (
        og_label_output_data <= choroid_mask_range[1]
    )
    og_color_retina = mask_to_color(
        og_label_output_data, og_retina_mask, ColorArray(retina_color)
    )
    og_color_choroid = mask_to_color(
        og_label_output_data, og_choroid_mask, ColorArray(choroid_color)
    )

    same_dims = np.ones(output_data.ndim, dtype=np.uint8)
    new_output_data = np.tile(output_data, (3, *same_dims))
    dims = np.arange(new_output_data.ndim)
    new_output_data = new_output_data.transpose(*dims[1:], 0)
    combo_label = color_retina + color_choroid
    combo_img = color_retina + color_choroid + new_output_data

    same_dims2 = np.ones(og_output_data.ndim, dtype=np.uint8)
    new_output_data2 = np.tile(og_output_data, (3, *same_dims2))
    dims2 = np.arange(new_output_data2.ndim)

    # new_output_data2 = new_output_data2.transpose(-2,-1,-3)

    print("same_dims", same_dims, same_dims2)
    print("new output data shapes", new_output_data.shape, new_output_data2.shape)
    print("dims", dims, dims2, (*dims[1:], 0), (*dims2[1:], 0))

    new_output_data2 = new_output_data2.transpose(*dims2[1:], 0)
    og_combo_label = og_color_retina + og_color_choroid
    og_combo_img = og_color_retina + og_color_choroid + new_output_data2

    viewer.add_image(original_shape_data, name="original_shape")
    viewer.add_image(original_label_shape_data, name="orginal_labels_image")
    viewer.add_image(og_output_data, name="og_output_data")
    viewer.add_image(og_label_output_data, name="og_label_output_data")
    viewer.add_labels(og_retina_mask * 10, name="og_retina_mask")
    viewer.add_labels(og_choroid_mask * 6, name="og_choroid_mask")
    viewer.add_image(og_combo_img, name="no_curve_comgo_image")
    viewer.add_image(output_data, name="uncle_ben-s_2D_curve_correction")
    viewer.add_image(label_output_data, name="uncle_ben-s_2D_cure_correction_lbls")
    viewer.add_labels(retina_mask * 10, name="retina_mask")
    viewer.add_labels(choroid_mask * 6, name="choroid_mask")
    # viewer.add_image(color_retina)
    # viewer.add_image(color_choroid)
    viewer.add_image(combo_label)
    viewer.add_image(combo_img)
    # viewer.add_image(new_output_data)

    last_layer = viewer.layers[-1]

    if use_color_map:
        color_0 = ColorArray((0.0, 0.0, 0.0))
        # color_1 = ColorArray((0.5,0.0,0.0))
        color_1 = ColorArray((0.0, 0.0, 0.0))
        color_2 = ColorArray(retina_color)  # ColorArray((0.5,0.0,0.0))
        color_3 = ColorArray(choroid_color)  # ColorArray((0.0,0.0,0.5))
        color_4 = ColorArray(choroid_color)  # ColorArray((0.0,0.0,0.5))
        # color_2 = ColorArray((0.0,0.5,0.0))
        color_map = Colormap(
            colors=(color_0, color_1, color_2, color_3, color_4),
            controls=(0.0, 0.47, 0.471, 0.472, 1.0),
            interpolation="linear",
        )
        last_layer.colormap = color_map

    # viewer.add_labels(cart,name="uncle_ben-s_curve_correction")
    viewer.show()
    napari.run()


generate_fast_curve_correction.show(run=True)
