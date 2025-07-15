import lightning as L
import napari

# from pathlib import Path
import numpy as np
import torch
from jj_nn_framework.data_setup import LoadNumpyData
from jj_nn_framework.nn_transforms import Normalize, PadToTargetM, ResizeToFit
from jj_nn_framework.yakub_complex_conjugate_unet import UNet
from napari_cool_tools_io import torch
from napari_cool_tools_segmentation import (
    Path,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


class WrappedInLightning(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        return self.model(input)


test_volume = Path(
    r"F:\Registration Sample\Test_Data\All_Peripheral_for registration\2024.09.18\08983951\08983951_13_09_18_processed.prof"
)
target_shape = (512, 1024)
processor = "cuda"
batch_size = 8
num_workers = 0

viewer = napari.Viewer(show=False)
viewer.open(test_volume, plugin="napari-cool-tools-io")
img = viewer.layers[-1].data.transpose((-3, -1, -2))

# meta = prof_proc_meta(test_volume, ".prof")
# h, w, d, bmscan, w_param, dtype, layer_type = meta

# dot_prof = np.dtype(("<f4", (h, w)))
# img = np.fromfile(test_volume, dtype=dot_prof, count=-1)

# viewer.add_image(img,name="original")

pttm_params = {
    "h": target_shape[-2],  # 992 #256 512, 992, 864, 800,
    "w": target_shape[-1],  # 800 #224 416, 800, 864, 800,
    "X_data_format": "NHW",  #'HW','NHW','NCHW',
    "y_data_format": "NHW",  #'HW','NHW', 'NCHW',
    "mode": "constant",
    "value": None,
    "pad_gt": False,
    "device": processor,
}
init_shape = (img.shape[-2], img.shape[-1])

pred_trans = torch.nn.Sequential(
    ResizeToFit(target_shape),
    PadToTargetM(**pttm_params),
    Normalize(),  # Standardize(),Normalize(),
)

pred_ds = LoadNumpyData(
    img,
    chunk_size=batch_size,
    transform=pred_trans,
    preprocessing=None,
    device=processor,
)

pred_dl = DataLoader(
    pred_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

test_volume = Path(
    r"F:\Registration Sample\Test_Data\All_Peripheral_for registration\2024.09.18\08983951\08983951_13_09_18_processed.prof"
)

checkpoint_path = Path(
    r"D:\JJ\Development\Yakub_Complex_Conjugate_Processing\U-net_resources\unet_denoise_epoch201.pth"
)
onnx_model_path = Path(
    r"D:\JJ\Development\Yakub_Complex_Conjugate_Processing\U-net_resources\onnx"
)
# onnx_model_path = Path(r"./")
onnx_model_path.mkdir(parents=False, exist_ok=True)
# model_paths = list(model_path.glob('*.pth'))
# model_path = list(weight_path.glob('*.pth'))[0]
model_state_dict = torch.load(checkpoint_path, weights_only=True)["model_state_dict"]
yakub_model = UNet(in_channels=1, out_channels=1)
yakub_model.load_state_dict(model_state_dict)
yakub_model.cuda()
yakub_lit_model = WrappedInLightning(yakub_model)

# trainer = L.Trainer()
# trainer.test(yakub_lit_model,pred_dl)

output = []
for image_batch in tqdm(pred_dl, desc="Removing complex conjugate from B-scans:"):
    # if idx == 60:
    # viewer.add_image(image_batch.detach().cpu().numpy(),name="test_batch")
    result = yakub_lit_model(image_batch)
    # print(result.shape)
    # viewer.add_image(result.cpu().detach().numpy(),name="clean?")
    output.append(result.cpu().detach().numpy())
    # break
cleaned = np.concatenate(output, axis=0).squeeze()
# print(cleaned.shape)
viewer.add_image(
    cleaned.transpose((-3, -1, -2)), name=f"{viewer.layers[0].name}_deconjugated"
)

viewer.show()
napari.run()
