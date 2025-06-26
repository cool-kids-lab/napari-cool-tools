import pathlib
from pathlib import Path

import napari
import torch
from magicgui import magicgui

# import napari_cool_tools_io
from tqdm import tqdm

# from napari_cool_tools_oct_preproc import _oct_preproc_utils #generate_enface_image


@magicgui(
    data={"label": "B-scan variations", "mode": "r"},
    call_button="Display B-scan Variants",
)
def view_bscan_variants(
    data: pathlib.Path = Path(
        r"D:\Mani\BScan Labels 24\Done\Pytorch_Unet\Base_Data\output.pt"
    ),
):
    """"""
    in_dict = torch.load(data)

    viewer = napari.Viewer()

    images = in_dict["images"]
    labels = in_dict["masks"]
    names = in_dict["names"]

    print(f"images shape: {images.shape}, len names: {len(names)}\n")

    for i, name in tqdm(enumerate(names), desc="Load B-scan variations"):
        if i == len(names) - 1:
            viewer.add_labels(labels, name=name)
        else:
            viewer.add_image(images[i], name=name)


# view_bscan_variants.changed.connect(print)
view_bscan_variants.show(run=True)
