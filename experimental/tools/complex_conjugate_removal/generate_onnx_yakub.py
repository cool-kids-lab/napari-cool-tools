""" """

from pathlib import Path

import lightning as L
import torch
from jj_nn_framework.yakub_complex_conjugate_unet import UNet


# create lightning module version of trained model to save as onnx file
class WrappedInLightning(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        return self.model(input)


# model_path = Path(r"D:\\JJ\\Development\Aaron_UNET_Mani_Images-Refactor2\\out_pl_beth_models\\to_process")
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
yakub_lit_model = WrappedInLightning(yakub_model)

onnx_model_out_path = onnx_model_path / "conjugate_clean.onnx"
dummy_input = torch.randn(1, 1, 512, 1024)
# kwargs = {'dynamic_axes':{'model_input':{0:'batchsize'}}}
kwargs = {
    "input_names": ["input"],
    "output_names": ["output"],
    #'do_constant_folding': True,
    "dynamic_axes": {"input": {0: "batchsize"}, "output": {0: "batchsize"}},
}
yakub_lit_model.to_onnx(onnx_model_out_path, dummy_input, **kwargs)
