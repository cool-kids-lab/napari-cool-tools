"""
This module contains code for segmenting images
"""
import gc
import numpy as np
import onnxruntime as onrt
from pathlib import Path
from typing import List, Tuple, Generator, Literal
from magicgui import magic_factory
from tqdm import tqdm
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker
from napari.layers import Image, Layer
from napari.types import ImageData
from napari_cool_tools_io import torch,viewer,device,memory_stats
from napari_cool_tools_segmentation._segmentation_funcs import bscan_onnx_seg_func
from napari_cool_tools_segmentation._segmentation_funcs import enface_onnx_seg_func

this_file_path = Path(__file__)
onnx_folder_parent_path = this_file_path.parents[3]
onnx_bscan_path = onnx_folder_parent_path / "onnx_models/bscan/"
onnx_enface_path = onnx_folder_parent_path / "onnx_models/enface"
onnx_enface_vessels_path = onnx_enface_path / "vessels"
onnx_enface_optic_nerve_path = onnx_enface_path / "optic_nerve"
print(onnx_bscan_path)
print(list(onnx_bscan_path.rglob("*.onnx")))
onnx_bscan = list(onnx_bscan_path.rglob("*.onnx"))[0]
onnx_enface_vessels = list(onnx_enface_vessels_path.rglob("*.onnx"))[0]
onnx_enface_optic_nerve = list(onnx_enface_optic_nerve_path.rglob("*.onnx"))[0]

@magic_factory()
def bscan_onnx_seg_plugin(img:Image, 
                          #onnx_path=Path("../onnx_models/bscan/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856.onnx"),
                          onnx_path=onnx_bscan,
                          batch_size:int=32, num_workers:int=0,
                          use_cpu:bool=False,output_preproc:bool=False,old_preproc:bool=False,debug:bool=False):
    """"""
    bscan_onnx_seg_thread(img,onnx_path=onnx_path,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          use_cpu=use_cpu,
                          output_preproc=output_preproc,
                          old_preproc=old_preproc,
                          debug=debug)
    return

#@thread_worker(connect={"returned": viewer.add_layer})
@thread_worker(connect={"yielded": viewer.add_layer})
def bscan_onnx_seg_thread(img:Image, 
                          #onnx_path=Path("../onnx_models/bscan/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856.onnx"),
                          onnx_path=onnx_bscan,
                          batch_size:int=32, num_workers:int=0,
                          use_cpu:bool=True,output_preproc:bool=False,old_preproc:bool=False,debug:bool=False):
    """"""
    show_info("Onnx B-scan thread has started\n")

    labels_name = f"{img.name}_B-scan_labels"
    preproc_name = f"{img.name}_B-scan_preproc"
    
    outputs = bscan_onnx_seg_func(img.data,
                                  onnx_path=onnx_path,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  use_cpu=use_cpu,
                                  output_preproc=output_preproc,
                                  old_preproc=old_preproc,
                                  debug=debug)

    for layer,layer_type in  outputs:

        add_kwargs = {}

        if layer_type == "labels":
            add_kwargs["name"] = labels_name
            
        elif layer_type == "image":
            add_kwargs["name"] = preproc_name
        
        out_layer = Layer.create(layer,add_kwargs,layer_type)
        yield(out_layer)

    show_info("Onnx B-scan thread has completed\n")
    #yield out_layer


@magic_factory()
def enface_popcorn_seg_func(img:Image, state_dict_path=Path("../nn_state_dicts/enface/Popcorn_model_best_iou_06.pth"), threshold:float = 0.6, label:int=2, use_cpu:bool=True, output_preproc:bool=False) -> List[Layer]:
    """
    """
    from napari_cool_tools_io import device
    from torchvision.transforms import v2
    from segmentation_models_pytorch import Unet
    from kornia.enhance import equalize_clahe, adjust_log
    from jj_nn_framework.image_funcs import normalize_in_range, pad_to_target_2d, pad_to_targetM_2d, bw_1_to_3ch

    layers_out = []

    target_size = (800,832)

    if use_cpu:
        device = 'cpu'

    pttm_params = {
        'h': 800,
        'w': 832,
        'X_data_format': 'NCHW',
        'y_data_format': 'NHW',
        'mode': 'constant',
        'value': None,
        'device': device
    }

    # get data
    data = img.data.copy()

    og_size = (data.shape[-2],data.shape[-1])

    pt_data = torch.tensor(data,device=device)
    #print(f"pt_data shape: {pt_data.shape}\n")
    ch3_data = bw_1_to_3ch(pt_data,data_format='HW')
    #print(f"ch3_data shape: {ch3_data.shape}\n")
    norm_ch3_data = normalize_in_range(ch3_data,0.0,1.0)

    # resize data
    resizer = v2.Resize(target_size)
    x = resizer(norm_ch3_data)

    # preproc data
    mean,std = x.mean([0,2,3]),x.std([0,2,3])
    #norm = v2.Normalize(mean,std)
    norm = v2.Normalize(mean=[0.485],std=[0.229])
    x_norm = norm(x)
    x_norm2 = normalize_in_range(x_norm,0,1)
    print("\n\nx_nomr min/max:",x_norm2.min(),x_norm2.max(),"\n\n")
    #x_eq = equalize_clahe(x_norm2,clip_limit=3.0)
    x_eq = x_norm2
    x_preproc = adjust_log(x_eq,gain=1)
    x_norm3 = normalize_in_range(x_preproc,0,255)
    x_preproc = x_norm3

    # Load the model
    model = Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.eval()
    model.to(device)
    print(model)

    with torch.no_grad():
        pred = model(x_preproc)
        print(pred.min(),pred.max())
        pred = torch.sigmoid(pred)
        print(pred.min(),pred.max())
        print(len(pred.nonzero()))
        pred_out = (pred > threshold)
        pred_out = pred_out.to(torch.bool).to(torch.uint8)*label
        #pred_out = pred.squeeze().cpu().numpy().astype(np.uint8)
        og_sizer = v2.Resize(og_size)
        pred_out = og_sizer(pred_out).squeeze().cpu().numpy()
        #pred_out = pred.squeeze().cpu().numpy()
        print(pred_out.shape)

    #layers_out.append(img)

    if output_preproc:
        name = f"{img.name}_Popcorn_preproc"
        add_kwargs = {"name":f"{name}"}
        layer_type = "image" #"labels"
        layer = Layer.create(x_eq.cpu().numpy(),add_kwargs,layer_type)
        layers_out.append(layer)

    name = f"{img.name}_Popcorn"
    add_kwargs = {"name":f"{name}"}
    layer_type = "labels"
    layer = Layer.create(pred_out,add_kwargs,layer_type)
    layers_out.append(layer)

    return layers_out


@magic_factory()
def enface_onnx_seg_plugin(
        img:Image,
        segmentation:Literal["optic_nerve","vessel"] = "vessel",
        #onnx_path=Path("../onnx_models/enface/UWF_OCT_enface_seg_EP_200_PR_16-mixed_SD_60_05-10-2024_12h50m_every_10-epoch=0069-step=3430.onnx"),
        label_val:int=1,
        use_cpu:bool=True,
        DoG:bool=False,
        blur:bool=False,
        log_adjust:bool=False,
        output_preproc:bool=False,
        debug:bool=False
    ) -> List[Layer]:
    """Function runs image/volume through pixwpixHD trained generator network to create segmentation labels. 
    Args:
        img (Image): Image/Volume to be segmented.
        state_dict_path (Path): Path to state dictionary of the network to be used for inference.
        label_flag (bool): If true return labels layer with relevant masks as unique label values
                           If false returns volume with unique channels masked with value 1.
        
    Yields:
        Image Layer containing padded enface image with '_Pad' suffix added to name
        Labels Layer containing B-scan segmentations with '_Seg' suffix added to name.
    """
    from napari_cool_tools_io import device
    from napari_cool_tools_img_proc._equalization_funcs import normalize_data_in_range_pt_func
    from jj_nn_framework.image_funcs import normalize_in_range, pad_to_target_2d, pad_to_targetM_2d, bw_1_to_3ch
    from jj_nn_framework.nn_transforms import DiffOfGausPred
    from torchvision.transforms import v2
    from kornia.enhance import equalize_clahe, adjust_log
    from kornia.filters import gaussian_blur2d
    from onnxruntime import InferenceSession

    #r"D:\\JJ\Development\\COOL_Tools_plugin\\onnx_models\\UWF_OCT_enface_seg_EP_200_PR_16-mixed_SD_60_05-10-2024_12h50m_every_10-epoch=0069-step=3430.onnx"
    if segmentation == "vessel":
        onnx_path = onnx_enface_vessels
    elif segmentation == "optic_nerve":
        onnx_path = onnx_enface_optic_nerve

    layers_out = []

    if use_cpu:
        device = 'cpu'

    pad_flag = False
    resize_flag = False

    pttm_params = {
        'h': 864,
        'w': 864,
        'X_data_format': 'NCHW',
        'y_data_format': 'NHW',
        'mode': 'constant',
        'value': None,
        'device': device
    }

    dog_params = {
        'low_sigma': 0.5, #0.0, #1.0,
        'high_sigma': 6.0, #20.0,
        'truncate': 4.0,
        'gamma': 1.0, #1.2,
        'gain': 1.0
    }

    final_seg = enface_onnx_seg_func(img.data,onnx_path=onnx_path,label_val=label_val,use_cpu=use_cpu,DoG=DoG,blur=blur,log_adjust=log_adjust,output_preproc=output_preproc,debug=debug)

    # data = img.data.copy()

    # if data.dtype == 'float64':
    #     data = data.astype('float32')
    # elif data.dtype == 'uint8':
    #     #data = normalize_data_in_range_pt_func()
    #     data = normalize_in_range(data.astype('float32'),min_val=0.0,max_val=1.0)
    # elif data.dtype != 'float32':
    #     ValueError(f"{data.dtype} is not supported float32, float64, and uint8 are supported")

    # pt_data = torch.tensor(data,device=device)
    # #print(f"pt_data shape: {pt_data.shape}\n")
    # ch3_data = bw_1_to_3ch(pt_data,data_format='HW')
    # #print(f"ch3_data shape: {ch3_data.shape}\n")
    # norm_ch3_data = normalize_in_range(ch3_data,0.0,1.0)
    # #print(f"norm_ch3_data shape: {norm_ch3_data.shape}\n")


    # if norm_ch3_data.shape[-1] < 864 and norm_ch3_data.shape[-2] < 864:
    #     pad_flag = True
    #     mod_data = pad_to_targetM_2d(norm_ch3_data,(864,864),'NCHW')
    #     print(f"pad_flag (shape): {mod_data.shape}\n")
    # elif norm_ch3_data.shape[-1] > 864 or norm_ch3_data.shape[-2] > 864:
    #     resize_flag = True
    #     original_shape = (norm_ch3_data.shape[-2],norm_ch3_data.shape[-1])
    #     mod_data = v2.functional.resize(norm_ch3_data,(864,864),interpolation=v2.InterpolationMode.BICUBIC)
    #     print(f"resize_flag (shape): {mod_data.shape}\n")
    # else:
    #     mod_data = norm_ch3_data


    # #pad_data = pad_to_targetM_2d(norm_ch3_data,(864,864),'NCHW')

    # out = mod_data.detach().cpu().numpy().squeeze()

    # if debug == True:
    #     name = f"{img.name}_Pad"
    #     add_kwargs = {"name":f"{name}"}
    #     layer_type = "image"
    #     layer_padded = Layer.create(out,add_kwargs,layer_type)
    #     layers_out.append(layer_padded)

    # if pad_flag:
    #     offset_0 = out[0].shape[0] - data.shape[0]
    #     offset_1 = out[0].shape[1] - data.shape[1]
    #     start_0 = int(offset_0/2)
    #     start_1 = int(offset_1/2)
    #     end_0 = int(out[0].shape[0] - start_0)
    #     end_1 = int(out[0].shape[1] - start_1)

    # x = normalize_in_range(mod_data,0,1)
    # mean,std = x.mean([0,2,3]),x.std([0,2,3])
    # norm = v2.Normalize(mean,std)
    # x_norm = norm(x)
    # x_norm2 = normalize_in_range(x_norm,0,1)

    # #x_eq = equalize_clahe(x_norm2)
    # x_eq = equalize_clahe(x_norm2,clip_limit=3.0)

    # if log_adjust == True:
    #     #x_eq = adjust_log(x,gain=1)
    #     x_eq = adjust_log(x_eq,gain=1)

    # if DoG == True:
    #     diff_of_gauss = DiffOfGausPred(**dog_params)
    #     x_eq = diff_of_gauss(x_eq)

    # if blur == True:
    #     x_eq = gaussian_blur2d(x_eq,kernel_size=3,sigma=(1.0,1.0),border_type='reflect')
    #     #x = normalize_in_range(x_eq,0,1)
    #     x_eq = normalize_in_range(x_eq,0,1)

    # #print(f"x shape: {x_eq.shape}\n")

    # ENCODER = "efficientnet-b5"
    # ENCODER_WEIGHTS = "imagenet"
    # CLASSES = [
    #     "vessel"
    # ]
    # ACTIVATION = "sigmoid"

    # '''
    # model = smp.Unet(encoder_name=ENCODER, # smp.UnetPlusPlus(encoder_name=ENCODER,
    #                 encoder_weights=ENCODER_WEIGHTS,
    #                 classes=len(CLASSES),
    #                 activation=ACTIVATION)
    # state_dict = torch.load(state_dict_path,map_location=device)
    # model.load_state_dict(state_dict)
    # model.eval()
    # model_dev = model.to(device)
    # output = model_dev.predict(x_eq)
    # '''

    # #show_info(f'x_eq: {x_eq.shape}')

    # x_eq_cpu = x_eq.detach().cpu().numpy()
    # pre_poc = x_eq.mean(dim=0).detach().cpu() #.numpy()

    # if pad_flag:
    #     pre_proc_final = pre_poc[start_0:end_0,start_1:end_1]
    # elif resize_flag:
    #     pre_proc_final = v2.functional.resize(pre_poc,original_shape,interpolation=v2.InterpolationMode.BICUBIC)
    # else:
    #     pre_proc_final = pre_poc

    # if output_preproc == True:
    #     name = f"{img.name}_Preproc"
    #     add_kwargs = {"name":f"{name}"}
    #     layer_type = "image"

    #     layer = Layer.create(pre_proc_final.squeeze().numpy(),add_kwargs,layer_type)
    #     layers_out.append(layer)

    # # start onnx
    # onnx_session = InferenceSession(onnx_path)
    # input_name = onnx_session.get_inputs()[0].name

    # onnx_inputs = {input_name: x_eq_cpu}
    # onnx_outs = onnx_session.run(None, onnx_inputs)
    # onnx_out = onnx_outs[0].squeeze().astype(np.uint8)
    
    # #seg_out = onnx_out.detach().cpu().numpy().squeeze().astype(int)

    # if pad_flag:
    #     final_seg = onnx_out[start_0:end_0,start_1:end_1].astype(bool)*(label_val)
    # elif resize_flag:
    #     final_seg = v2.functional.resize(torch.tensor(onnx_out).unsqueeze(0),original_shape,v2.InterpolationMode.NEAREST_EXACT).numpy().astype(bool)*(label_val)
    # else:
    #     final_seg = onnx_out.astype(bool)*(label_val)

    name = f"{img.name}_Seg"
    add_kwargs = {"name":f"{name}"}
    layer_type = "labels"
    layer = Layer.create(final_seg,add_kwargs,layer_type)

    #viewer.add_layer(layer)

    layers_out.append(layer)
    
    # # clean up
    del final_seg
    # del final_seg, onnx_session
    # #del seg_out
    # #del output
    # #del model_dev
    # #del model
    # del x_eq
    # del x_norm2
    # del x_norm
    # del norm
    # del mean
    # del std
    # del x
    # del out
    # del mod_data
    # del norm_ch3_data
    # del ch3_data
    # del pt_data

    gc.collect()
    torch.cuda.empty_cache()

    return layers_out



# @magic_factory()
# def enface_onnx_seg_plugin(
#     img:Image,
#     segmentation:Literal["optic_nerve","vessel"],
#     #onnx_path=Path("../onnx_models/enface/UWF_OCT_enface_seg_EP_200_PR_16-mixed_SD_60_05-10-2024_12h50m_every_10-epoch=0069-step=3430.onnx"),
#     label_val:int=1,
#     use_cpu:bool=True,
#     DoG:bool=False,
#     blur:bool=False,
#     log_adjust:bool=False,
#     output_preproc:bool=False,
#     debug:bool=False
# ) -> List[Layer]:
#     """Function runs image/volume through pixwpixHD trained generator network to create segmentation labels. 
#     Args:
#         img (Image): Image/Volume to be segmented.
#         state_dict_path (Path): Path to state dictionary of the network to be used for inference.
#         label_flag (bool): If true return labels layer with relevant masks as unique label values
#                            If false returns volume with unique channels masked with value 1.
        
#     Yields:
#         Image Layer containing padded enface image with '_Pad' suffix added to name
#         Labels Layer containing B-scan segmentations with '_Seg' suffix added to name.
#     """
#     from napari_cool_tools_io import device
#     from napari_cool_tools_img_proc._equalization_funcs import normalize_data_in_range_pt_func
#     from jj_nn_framework.image_funcs import normalize_in_range, pad_to_target_2d, pad_to_targetM_2d, bw_1_to_3ch
#     from jj_nn_framework.nn_transforms import DiffOfGausPred
#     from torchvision.transforms import v2
#     from kornia.enhance import equalize_clahe, adjust_log
#     from kornia.filters import gaussian_blur2d
#     from onnxruntime import InferenceSession

#     #r"D:\\JJ\Development\\COOL_Tools_plugin\\onnx_models\\UWF_OCT_enface_seg_EP_200_PR_16-mixed_SD_60_05-10-2024_12h50m_every_10-epoch=0069-step=3430.onnx"
#     if segmentation == "vessel":
#         onnx_path = onnx_enface_vessels
#     elif segmentation == "optic_nerve":
#         onnx_path = onnx_enface_optic_nerve

#     layers_out = []

#     if use_cpu:
#         device = 'cpu'

#     pad_flag = False
#     resize_flag = False

#     pttm_params = {
#         'h': 864,
#         'w': 864,
#         'X_data_format': 'NCHW',
#         'y_data_format': 'NHW',
#         'mode': 'constant',
#         'value': None,
#         'device': device
#     }

#     dog_params = {
#         'low_sigma': 0.5, #0.0, #1.0,
#         'high_sigma': 6.0, #20.0,
#         'truncate': 4.0,
#         'gamma': 1.0, #1.2,
#         'gain': 1.0
#     }

#     data = img.data.copy()

#     if data.dtype == 'float64':
#         data = data.astype('float32')
#     elif data.dtype == 'uint8':
#         data = normalize_in_range(data.astype('float32'),min_val=0.0,max_val=1.0)
#     elif data.dtype != 'float32':
#         ValueError(f"{data.dtype} is not supported float32, float64, and uint8 are supported")

#     pt_data = torch.tensor(data,device=device)
#     #print(f"pt_data shape: {pt_data.shape}\n")
#     ch3_data = bw_1_to_3ch(pt_data,data_format='HW')
#     #print(f"ch3_data shape: {ch3_data.shape}\n")
#     norm_ch3_data = normalize_in_range(ch3_data,0.0,1.0)
#     #print(f"norm_ch3_data shape: {norm_ch3_data.shape}\n")


#     if norm_ch3_data.shape[-1] < 864 and norm_ch3_data.shape[-2] < 864:
#         pad_flag = True
#         mod_data = pad_to_targetM_2d(norm_ch3_data,(864,864),'NCHW')
#         print(f"pad_flag (shape): {mod_data.shape}\n")
#     elif norm_ch3_data.shape[-1] > 864 or norm_ch3_data.shape[-2] > 864:
#         resize_flag = True
#         original_shape = (norm_ch3_data.shape[-2],norm_ch3_data.shape[-1])
#         mod_data = v2.functional.resize(norm_ch3_data,(864,864),interpolation=v2.InterpolationMode.BICUBIC)
#         print(f"resize_flag (shape): {mod_data.shape}\n")
#     else:
#         mod_data = norm_ch3_data


#     #pad_data = pad_to_targetM_2d(norm_ch3_data,(864,864),'NCHW')

#     out = mod_data.detach().cpu().numpy().squeeze()

#     if debug == True:
#         name = f"{img.name}_Pad"
#         add_kwargs = {"name":f"{name}"}
#         layer_type = "image"
#         layer_padded = Layer.create(out,add_kwargs,layer_type)
#         layers_out.append(layer_padded)

#     if pad_flag:
#         offset_0 = out[0].shape[0] - data.shape[0]
#         offset_1 = out[0].shape[1] - data.shape[1]
#         start_0 = int(offset_0/2)
#         start_1 = int(offset_1/2)
#         end_0 = int(out[0].shape[0] - start_0)
#         end_1 = int(out[0].shape[1] - start_1)

#     x = normalize_in_range(mod_data,0,1)
#     mean,std = x.mean([0,2,3]),x.std([0,2,3])
#     norm = v2.Normalize(mean,std)
#     x_norm = norm(x)
#     x_norm2 = normalize_in_range(x_norm,0,1)

#     #x_eq = equalize_clahe(x_norm2)
#     x_eq = equalize_clahe(x_norm2,clip_limit=3.0)

#     if log_adjust == True:
#         #x_eq = adjust_log(x,gain=1)
#         x_eq = adjust_log(x_eq,gain=1)

#     if DoG == True:
#         diff_of_gauss = DiffOfGausPred(**dog_params)
#         x_eq = diff_of_gauss(x_eq)

#     if blur == True:
#         x_eq = gaussian_blur2d(x_eq,kernel_size=3,sigma=(1.0,1.0),border_type='reflect')
#         #x = normalize_in_range(x_eq,0,1)
#         x_eq = normalize_in_range(x_eq,0,1)

#     #print(f"x shape: {x_eq.shape}\n")

#     ENCODER = "efficientnet-b5"
#     ENCODER_WEIGHTS = "imagenet"
#     CLASSES = [
#         "vessel"
#     ]
#     ACTIVATION = "sigmoid"

#     '''
#     model = smp.Unet(encoder_name=ENCODER, # smp.UnetPlusPlus(encoder_name=ENCODER,
#                     encoder_weights=ENCODER_WEIGHTS,
#                     classes=len(CLASSES),
#                     activation=ACTIVATION)
#     state_dict = torch.load(state_dict_path,map_location=device)
#     model.load_state_dict(state_dict)
#     model.eval()
#     model_dev = model.to(device)
#     output = model_dev.predict(x_eq)
#     '''

#     #show_info(f'x_eq: {x_eq.shape}')

#     x_eq_cpu = x_eq.detach().cpu().numpy()
#     pre_poc = x_eq.mean(dim=0).detach().cpu() #.numpy()

#     if pad_flag:
#         pre_proc_final = pre_poc[start_0:end_0,start_1:end_1]
#     elif resize_flag:
#         pre_proc_final = v2.functional.resize(pre_poc,original_shape,interpolation=v2.InterpolationMode.BICUBIC)
#     else:
#         pre_proc_final = pre_poc

#     if output_preproc == True:
#         name = f"{img.name}_Preproc"
#         add_kwargs = {"name":f"{name}"}
#         layer_type = "image"

#         layer = Layer.create(pre_proc_final.squeeze().numpy(),add_kwargs,layer_type)
#         layers_out.append(layer)

#     # start onnx
#     onnx_session = InferenceSession(onnx_path)
#     input_name = onnx_session.get_inputs()[0].name

#     onnx_inputs = {input_name: x_eq_cpu}
#     onnx_outs = onnx_session.run(None, onnx_inputs)
#     onnx_out = onnx_outs[0].squeeze().astype(np.uint8)
    
#     #seg_out = onnx_out.detach().cpu().numpy().squeeze().astype(int)

#     if pad_flag:
#         final_seg = onnx_out[start_0:end_0,start_1:end_1].astype(bool)*(label_val)
#     elif resize_flag:
#         final_seg = v2.functional.resize(torch.tensor(onnx_out).unsqueeze(0),original_shape,v2.InterpolationMode.NEAREST_EXACT).numpy().astype(bool)*(label_val)
#     else:
#         final_seg = onnx_out.astype(bool)*(label_val)

#     name = f"{img.name}_Seg"
#     add_kwargs = {"name":f"{name}"}
#     layer_type = "labels"
#     layer = Layer.create(final_seg,add_kwargs,layer_type)

#     #viewer.add_layer(layer)

#     layers_out.append(layer)
    
#     # clean up
#     del final_seg, onnx_session
#     #del seg_out
#     #del output
#     #del model_dev
#     #del model
#     del x_eq
#     del x_norm2
#     del x_norm
#     del norm
#     del mean
#     del std
#     del x
#     del out
#     del mod_data
#     del norm_ch3_data
#     del ch3_data
#     del pt_data

#     gc.collect()
#     torch.cuda.empty_cache()

#     return layers_out