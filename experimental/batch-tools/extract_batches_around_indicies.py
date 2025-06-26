import pathlib
from pathlib import Path
from pickle import HIGHEST_PROTOCOL

# import napari_cool_tools_io
from typing import List

import numpy as np
import torch
from magicgui import magicgui
from tqdm import tqdm

# from napari_cool_tools_oct_preproc import _oct_preproc_utils #generate_enface_image


@magicgui(
    data={"label": "Extract batches around features", "mode": "r"},
    output_dir={"label": "Output Directory", "mode": "d"},
    call_button="Extract batches",
)
def extract_batches_around_incicies(
    data: pathlib.Path = Path(
        r"D:\Mani\BScan Labels 24\Done\Pytorch_Unet\Base_Data\output.pt"
    ),
    output_dir: pathlib.Path = Path(r"D:\Mani\BScan Labels 24\Done\Pytorch_Unet\Folds"),
    output_filename: str = "output.pt",
    indicies: List[int] = [156, 240, 298, 380],
    batch_size: int = 48,  # axis:int=1,
    include_start: bool = True,
    include_stop: bool = True,
):
    """"""
    in_dict = torch.load(data)

    # viewer = napari.Viewer()

    images = in_dict["images"]
    labels = in_dict["masks"]
    names = in_dict["names"]

    print(f"images shape: {images.shape}, len names: {len(names)}\n")

    out_images = []
    out_labels = []

    if include_start:
        out_images.append(images[:, 0:batch_size])
        out_labels.append(labels[0:batch_size])
        print(f"Extracting slices [{0}:{batch_size}]\n")

    for i in tqdm(indicies, desc="Exracting batch around index"):
        start = i - int(batch_size / 2 - 1)
        stop = i + int(batch_size / 2 + 1)
        print(f"Extracting slices [{start}:{stop}]\n")
        out_images.append(images[:, start:stop])
        out_labels.append(labels[start:stop])

    if include_stop:
        out_images.append(images[:, -batch_size:])
        out_labels.append(labels[-batch_size:])
        print(f"Extracting slices [{images.shape[1] - batch_size}:{images.shape[1]}]\n")

    out_images = np.concatenate(out_images, axis=1)
    out_labels = np.concatenate(out_labels, axis=0)

    print(f"output images shape: {out_images.shape}\n")
    print(f"output images shape: {out_labels.shape}\n")

    out_dict = {}
    out_dict["images"] = out_images
    out_dict["masks"] = out_labels
    out_dict["names"] = names

    torch.save(
        out_dict,
        Path(f"{output_dir}\{output_filename}"),
        pickle_protocol=HIGHEST_PROTOCOL,
    )

    print("Extracted batches saved\n")


# view_bscan_variants.changed.connect(print)
extract_batches_around_incicies.show(run=True)
