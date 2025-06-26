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
def view_training_folds(
    data: pathlib.Path = Path(
        r"D:\Mani\BScan Labels 24\Done\Pytorch_Unet\Folds\Combined\output.pt"
    ),
):
    """"""
    in_dict = torch.load(data)

    viewer = napari.Viewer()

    images = in_dict["images"]
    labels = in_dict["masks"]
    names = in_dict["names"]
    test_fold_path = in_dict["test_fold_path"]

    print(f"images shape: {images.shape}, len names: {len(names)}\n")
    print(f"Associated test fold: {test_fold_path}\n")

    for image in tqdm(images, desc="Load B-scan variations"):
        viewer.add_image(image)

    viewer.add_labels(labels)


# view_bscan_variants.changed.connect(print)
view_training_folds.show(run=True)
