""" """

from pathlib import Path
#from typing import Literal
from enum import Enum

import lightning as L
import torch
from magicgui import magicgui

from jj_nn_framework.yakub_complex_conjugate_unet import UNet
from jj_nn_framework.ROP_vessel_unet import ROPVesselSegUnet, rop_vessel_config
from jj_nn_framework.project_models import LitUnet, rop_bscan_config

class MODELTYPE(Enum):
    ROPBSCANSEG = LitUnet
    ROPVESSELSEG = ROPVesselSegUnet
    DECONJUGATE = UNet

class WrappedInLightning(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        return self.model(input)

@magicgui(
    # checkpoint_path={"checkpoint path": "path to saved model checkpoint", "mode": "r"},
    saved_model_path={"label":"Path to saved pytorch model", "mode": "r"},
    onnx_model_path={"label": "directory to save onnx file to", "mode": "d"},
    call_button="Generate Onnx",
)
def generate_onnx_file(
    checkpoint_path: Path = Path(r"D:\JJ\Development\fake_model_data.pt"),
    saved_model_path: Path = Path(r"D:\JJ\Development\fake_model_data.pt"),
    onnx_model_path: Path = Path(r"D:\JJ\Development\_onnx_stash"),
    onnx_file_name: str = "model.onnx",
    dummy_input_shape: list = (1, 1, 864, 864),
    model_type:MODELTYPE = MODELTYPE.ROPVESSELSEG,
):
    # checkpoint_path = Path(r"D:\JJ\Development\Yakub_Complex_Conjugate_Processing\U-net_resources\unet_denoise_epoch201.pth")
    # onnx_model_path = Path(r"D:\JJ\Development\Yakub_Complex_Conjugate_Processing\U-net_resources\onnx")
    
    if model_type == MODELTYPE.ROPBSCANSEG:
        config = rop_bscan_config
        current_lightning_model = MODELTYPE.ROPBSCANSEG.value.load_from_checkpoint(checkpoint_path=checkpoint_path,train_config=config,loss_metric=None,acc_metric=None)
    elif model_type == MODELTYPE.ROPVESSELSEG:
        config = rop_vessel_config
        current_lightning_model = MODELTYPE.ROPVESSELSEG.value.load_from_checkpoint(checkpoint_path=checkpoint_path,train_config=config,loss_metric=None,acc_metric=None)

    #model_state_dict = torch.load(checkpoint_path, weights_only=True)
    #current_model.load_state_dict(model_state_dict)
    else:
        #current_lightning_model = WrappedInLightning(current_model)
        #current_lightning_model.to("cpu")
        pass

    onnx_model_out_path = onnx_model_path / f"{onnx_file_name}.onnx"
    print(f"onnx save path: {onnx_model_out_path}")
    dummy_input = torch.randn(*dummy_input_shape)
    # kwargs = {'dynamic_axes':{'model_input':{0:'batchsize'}}}
    kwargs = {
        "input_names": ["input"],
        "output_names": ["output"],
        #'do_constant_folding': True,
        "dynamic_axes": {"input": {0: "batchsize"}, "output": {0: "batchsize"}},
    }
    current_lightning_model.to_onnx(onnx_model_out_path, dummy_input, **kwargs)
    print("ONNX conversion complete")


generate_onnx_file.show(run=True)
