"""
This module contains function code for segmenting images
"""
import gc
import platform
from pathlib import Path
from typing import List, Tuple

import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm

from napari.utils.notifications import show_info
from napari.layers import Image, Layer
from napari.types import ImageData
from napari_cool_tools_io import torch,viewer,device,memory_stats


this_file_path = Path(__file__)
onnx_folder_parent_path = this_file_path.parents[2]
onnx_folder_path = onnx_folder_parent_path / "onnx_models/bscan/"


def bscan_onnx_seg_func(img:ImageData,
                        #onnx_path=Path("../onnx_models/bscan/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856.onnx"),
                        onnx_path=onnx_folder_path,
                        batch_size:int=32, num_workers:int=0,
                        gpu_limit:int=6,
                        use_cpu:bool=True,output_preproc:bool=False,old_preproc:bool=False,debug:bool=False):
    """"""
    import onnxruntime
    import torch.nn as nn
    from torchvision.transforms.v2.functional import resize
    from torch.utils.data import DataLoader
    from torchvision.transforms.functional import InterpolationMode
    from jj_nn_framework.data_setup import LoadNumpyData
    from jj_nn_framework.nn_transforms import (PadToTargetM,IntTarget,BscanPreproc2,NormalizeCLAHE2,ResizeToFit)

    target_shape = (992,800)
    init_shape = (img.shape[-2],img.shape[-1])

    if use_cpu:
        processor = 'cpu'
        onnx_dev = 'cpu'
        print(f"Using device {platform.processor()}")
    else:
        processor = device
        onnx_dev = 'cuda'
        device_id = torch.cuda.current_device()
        print(f"Using device {torch.cuda.get_device_name(device_id)}\n")

    
    print(f"Onnx file_path: {onnx_path}\n")
    
    num_bscans = len(img)
    rem = num_bscans % batch_size
    if rem != 0:
        missing_bscans = batch_size - rem
        fill_shape = (missing_bscans,img.shape[1],img.shape[2])
        batch_fill = np.empty(fill_shape,dtype=img.dtype)
        img = np.concatenate([img,batch_fill])
    

    onnx_folder_path = Path(onnx_path).parents[0]

    print(f"onnx_folder_path: {onnx_folder_path}\n")

    pttm_params = {
        'h': target_shape[-2], #992 #256 512, 992, 864, 800,
        'w': target_shape[-1], #800 #224 416, 800, 864, 800,
        'X_data_format': 'NHW', #'HW','NHW','NCHW',
        'y_data_format': 'NHW', #'HW','NHW', 'NCHW',
        'mode': 'constant',
        'value': None,
        'pad_gt': False,
        'device': processor
    }

    norm_clahe = {
        "log_gain": 2.5,
        'clahe_clip_limit': 1.0,
    }

    bscan_preproc_params = {
        'log_gain': 2.5, 
        'clahe_clip_limit': 1.0,
        'b_blur_ks': (5,5),
        'b_blur_sc': 0.1,
        'b_blur_ss': (1.0,1.0),
        'b_blur_bt': 'reflect',
        'g_blur_ks': (5,5),
        'g_blur_s': (1.0,1.0),
        'g_blur_bt': 'reflect',
    }

    if old_preproc:
        pred_trans = nn.Sequential(
            PadToTargetM(**pttm_params),
            BscanPreproc2(**bscan_preproc_params),
        )
    else:
        pred_trans = nn.Sequential(
            ResizeToFit(target_shape),
            PadToTargetM(**pttm_params),
            NormalizeCLAHE2()
        )


    pred_ds = LoadNumpyData(img,chunk_size=batch_size,transform=pred_trans,preprocessing=None,device=processor)

    pred_dl = DataLoader(pred_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers)

    #bscan_preproc = BscanPreproc2(**bscan_preproc_params)
    #bscan_preproc.to(processor)

    #img_tensor = torch.tensor(img.copy())
    #target_idx = int(len(img_tensor) / 2)
    #offset = 20
    #img_tensor = img_tensor[target_idx-offset:target_idx+offset]
    #img_tensor.to(processor)

    if use_cpu:
        providers = ['CPUExecutionProvider',]
    else:
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': device_id,                                    # Select GPU to execute
                'trt_max_workspace_size': gpu_limit * 1024 * 1024 * 1024, # Set GPU memory usage limit
                'trt_fp16_enable': True,                           # Enable FP16 precision for faster inference
                "trt_engine_cache_enable": True, #True,
                "trt_engine_cache_path": onnx_folder_path,
                "trt_timing_cache_enable": True, #True,
                "trt_timing_cache_path": onnx_folder_path,
                #"user_compute_stream": str(torch.cuda.current_stream().cuda_stream)
                "user_compute_stream": str(torch.cuda.Stream().cuda_stream)
            }),
            ('CUDAExecutionProvider', {
                'device_id': device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': gpu_limit * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
                'cudnn_conv_use_max_workspace': '1',
                #"user_compute_stream": str(torch.cuda.current_stream().cuda_stream)
                "user_compute_stream": str(torch.cuda.Stream().cuda_stream)
            }),
            'CPUExecutionProvider',
        ]

    """
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 20 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
            'cudnn_conv_use_max_workspace': '1',
            #"user_compute_stream": str(torch.cuda.current_stream().cuda_stream)
        }),
        'CPUExecutionProvider',
    ]
    """

    onnx_session = onnxruntime.InferenceSession(onnx_path,providers=providers)

    CLASSES = ['vitreous','retina','choroid'] # replace with parameter

    preproc_bscans = []
    label_preds = []

    for image_batch in tqdm(pred_dl,desc="Segmenting B-scans:"):

        # bindtensors to onnx session
        binding = onnx_session.io_binding()

        images_tensor = image_batch.contiguous()
        it_shape = images_tensor.shape

        binding.bind_input(
            name='input',
            device_type=onnx_dev,#'cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(it_shape),
            buffer_ptr=images_tensor.data_ptr(),
        )

        pred_shape = (it_shape[0],len(CLASSES),it_shape[2],it_shape[3])
        pred_tensor = torch.empty(pred_shape, dtype=torch.float32, device=onnx_dev).contiguous() #'cuda:0').contiguous()
        binding.bind_output(
            name='output',
            device_type=onnx_dev, #'cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(pred_tensor.shape),
            buffer_ptr=pred_tensor.data_ptr(),
        )

        # run onnx with bidning
        onnx_session.run_with_iobinding(binding)

        #print(f"pred_tensor shape:{pred_tensor.shape}\n")
        #pred_tensor = pred_tensor.reshape(-1,pred_shape[2],pred_shape[3])
        #print(f"pred_tensor shape:{pred_tensor.shape}\n")

        labels = []
        
        for i,mask in enumerate(pred_tensor):

            label = torch.zeros_like(mask[0],dtype=torch.uint8)
            mask_argmax = mask.argmax(0)
            for i,m in enumerate(mask):
                label[mask_argmax == i] = i
            
            labels.append(label)

        #print(f"label shape: {labels[0].shape}\n")
        labels = torch.stack(labels,dim=0)

        pred_tensor = pred_tensor.detach().squeeze().cpu().numpy()
        labels = labels.detach().squeeze().cpu().numpy()
        #pred_tensor = labels
        image_batch = image_batch.detach().squeeze().cpu().numpy()


        image_batch = image_batch[:num_bscans]
        #pred_tensor = pred_tensor[:num_bscans]

        #print(f"labels shape: {labels.shape}, pred_tensor shape: {pred_tensor.shape}\n")


        preproc_bscans.append(image_batch)

        #label_preds.append(pred_tensor[:,1,:,:])
        label_preds.append(labels)

    
    gpu_mem_clear = (torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0)
    print(f"GPU memory is clear: {gpu_mem_clear}\n")

    del (
        pred_ds,
        pred_dl,
        #image_batch,
        images_tensor,
        label, mask_argmax,
        mask, m
    )
    gc.collect()
    torch.cuda.empty_cache()

    gpu_mem_clear = (torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0)

    print(f"GPU memory is clear: {gpu_mem_clear}\n")
    if not gpu_mem_clear:
        print(f"{torch.cuda.memory_summary()}\n")


    preproc_bscans = np.concatenate(preproc_bscans,axis=0) #torch.concat(preproc_bscans,dim=0).detach().squeeze().cpu().numpy()
    label_preds = np.concatenate(label_preds,axis=0)
    label_preds_out = label_preds[:num_bscans]
    reshaped_out = resize(torch.tensor(label_preds_out.copy()),(init_shape),interpolation=InterpolationMode.NEAREST_EXACT).numpy()
    #label_preds = np.stack(label_preds,axis=0)

    output = []

    if output_preproc:
        output.append((preproc_bscans[:num_bscans],'image'))

    #output.append((label_preds[:num_bscans],'labels'))
    output.append((reshaped_out,'labels'))

    return output

def b_scan_pix2pixHD_seg_func(img:Image, state_dict_path=Path("./nn_state_dicts/b-scan/latest_net_G.pth"), label_flag:bool=True):
    """Function runs image/volume through pixwpixHD trained generator network to create segmentation labels. 
    Args:
        img (Image): Image/Volume to be segmented.
        state_dict_path (Path): Path to state dictionary of the network to be used for inference.
        label_flag (bool): If true return labels layer with relevant masks as unique label values
                           If false returns volume with unique channels masked with value 1.
        
    Returns:
        Labels Layer containing B-scan segmentations with '_Seg' suffix added to name.
    """
    from models.pix2pixHD_model import InferenceModel
    model = InferenceModel()
    state_dict = torch.load(state_dict_path)

    from models.networks import define_G

    def_g_settings = {
        "input_nc": 3,
        "output_nc": 3,
        "ngf": 64,
        "netG": 'global',
        "n_downsample_global": 4,
        "n_blocks_global": 9,
        "n_local_enhancers": 1,
        "n_blocks_local": 3,
        "norm": 'instance',
        "gpu_ids": [0],
    }

    data = img.data.copy()

    try:
        assert data.ndim == 2 or data.ndim == 3, "Only works for data of 2 or 3 dimensions"
    except AssertionError as e:
        print("An error Occured:", str(e))
    else:

        pt_data = torch.tensor(data,device=device)
    
        gen = define_G(**def_g_settings)
        gen_dev = gen.to(device)
        gen_dev.load_state_dict(state_dict)
        gen_dev.eval()
        
        name = f"{img.name}_Seg"
        add_kwargs = {"name":f"{name}"}

        if data.ndim == 2:
            pt_data2 = pt_data.unsqueeze(0).repeat(3,1,1)
            output = gen(pt_data2)
            retina = output[0] == 1
            choroid = output[1] == 1
            sclera = output[2] == 1

            if label_flag:
                labels = torch.zeros_like(output[0])
                labels[retina] = 1
                labels[choroid] = 2
                labels[sclera] = 3
                labels = labels.to(torch.uint8)
                labels_out = labels.detach().cpu().numpy()
                layer_type = 'labels'
                layer = Layer.create(labels_out,add_kwargs,layer_type)

                #clean up
                del labels_out
                del labels
                
            else:
                output2 = output.detach().cpu().numpy()
                layer_type = 'image'                
                layer = Layer.create(output2,add_kwargs,layer_type)
                
                #clean up
                del output2

            #clean up
            del retina
            del choroid
            del sclera
            del output
            del pt_data2
        
        elif data.ndim == 3:
            outstack = []
            for i in tqdm(range(len(data)),desc="B-scan Seg"):

                temp_data = pt_data[i].unsqueeze(0).repeat(3,1,1)
                output = gen(temp_data)
                retina = output[0] == 1
                choroid = output[1] == 1
                sclera = output[2] == 1

                if label_flag:
                    labels = torch.zeros_like(output[0])
                    labels[retina] = 1
                    labels[choroid] = 2
                    labels[sclera] = 3

                    outstack.append(labels)
                    #clean up
                    del labels

                else:
                    outstack.append(output)
                    #clean up
                    del output

                #clean up
                del retina
                del choroid
                del sclera
                del output
                del temp_data
                
                #clear cache from the loop
                #gc.collect() 
                #torch.cuda.empty_cache()

            if label_flag:
                labels2 = torch.stack(outstack)
                labels2 = labels2.to(torch.uint8)
                labels_out = labels2.detach().cpu().numpy()
                layer_type = 'labels'
                layer = Layer.create(labels_out,add_kwargs,layer_type)

                # clean up
                del labels_out
                del labels2
                del outstack
            else:
                output2 = torch.stack(outstack)
                layer_type = 'image'
                layer = Layer.create(output2,add_kwargs,layer_type)

                # clean up
                del output2
                del outstack

        #clean up
        del pt_data
        del gen_dev
        del gen
        gc.collect() 
    
    return layer

def enface_unet_seg_func(img:Image, state_dict_path=Path("./nn_state_dicts/enface/unet_efficientnet-b5_imagenet_dc10_sd_60_lr_5e-04_40EP_BS_32_04-19-2023_17h10m.pth"), 
                         use_cpu:bool=True) -> List[Layer]:
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
    from jj_nn_framework.image_funcs import normalize_in_range, pad_to_target_2d, pad_to_targetM_2d, bw_1_to_3ch
    from torchvision import transforms
    from kornia.enhance import equalize_clahe

    layers_out = []

    if use_cpu:
        device = 'cpu'

    pttm_params = {
        'h': 864,
        'w': 864,
        'X_data_format': 'NCHW',
        'y_data_format': 'NHW',
        'mode': 'constant',
        'value': None,
        'device': device
    }

    data = img.data.copy()
    pt_data = torch.tensor(data,device=device)
    #print(f"pt_data shape: {pt_data.shape}\n")
    ch3_data = bw_1_to_3ch(pt_data,data_format='HW')
    #print(f"ch3_data shape: {ch3_data.shape}\n")
    norm_ch3_data = normalize_in_range(ch3_data,0.0,1.0)
    #print(f"norm_ch3_data shape: {norm_ch3_data.shape}\n")

    crop_flag = False
    resize_flag = False

    if data.shape[-1] < 864 or data.shape[-2] < 864:
        crop_flag = True
    else:
        resize_flag = True


    pad_data = pad_to_targetM_2d(norm_ch3_data,(864,864),'NCHW')


    name = f"{img.name}_Pad"
    add_kwargs = {"name":f"{name}"}
    layer_type = "image"

    out = pad_data.detach().cpu().numpy().squeeze()

    offset_0 = out[0].shape[0] - data.shape[0]
    offset_1 = out[0].shape[1] - data.shape[1]
    start_0 = int(offset_0/2)
    start_1 = int(offset_1/2)
    end_0 = int(out[0].shape[0] - start_0)
    end_1 = int(out[0].shape[1] - start_1)

    x = normalize_in_range(pad_data,0,1)
    mean,std = x.mean([0,2,3]),x.std([0,2,3])
    norm = transforms.Normalize(mean,std)
    x_norm = norm(x)
    x_norm2 = normalize_in_range(x_norm,0,1)

    x_eq = equalize_clahe(x_norm2)

    print(f"x shape: {x_eq.shape}\n")

    name = f"{img.name}_Seg"
    add_kwargs = {"name":f"{name}"}
    layer_type = "labels"

    ENCODER = "efficientnet-b5"
    ENCODER_WEIGHTS = "imagenet"
    CLASSES = [
        "vessel"
    ]
    ACTIVATION = "sigmoid"

    model = smp.Unet(encoder_name=ENCODER, # smp.UnetPlusPlus(encoder_name=ENCODER,
                    encoder_weights=ENCODER_WEIGHTS,
                    classes=len(CLASSES),
                    activation=ACTIVATION)
    state_dict = torch.load(state_dict_path,map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model_dev = model.to(device)
    output = model_dev.predict(x_eq)
    
    seg_out = output.detach().cpu().numpy().squeeze().astype(int)
    final_seg = seg_out[start_0:end_0,start_1:end_1]
    layer = Layer.create(final_seg,add_kwargs,layer_type)
    
    # clean up
    del final_seg
    del seg_out
    del output
    del model_dev
    del model
    del x_eq
    del x_norm2
    del x_norm
    del norm
    del mean
    del std
    del x
    del out
    del pad_data
    del norm_ch3_data
    del ch3_data
    del pt_data

    gc.collect()

    layers_out.append(layer)
    return layers_out

def enface_onnx_seg_func(img:Image, onnx_path=Path("../onnx_models/enface/UWF_OCT_enface_seg_EP_200_PR_16-mixed_SD_60_05-10-2024_12h50m_every_10-epoch=0069-step=3430.onnx"), 
                         use_cpu:bool=True,DoG:bool=False,blur:bool=False,log_adjust:bool=False,output_preproc:bool=False,debug:bool=False) -> List[Layer]:
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
    from jj_nn_framework.image_funcs import normalize_in_range, pad_to_target_2d, pad_to_targetM_2d, bw_1_to_3ch
    from jj_nn_framework.nn_transforms import DiffOfGausPred
    from torchvision.transforms import v2
    from kornia.enhance import equalize_clahe, adjust_log
    from kornia.filters import gaussian_blur2d
    from onnxruntime import InferenceSession

    #r"D:\\JJ\Development\\COOL_Tools_plugin\\onnx_models\\UWF_OCT_enface_seg_EP_200_PR_16-mixed_SD_60_05-10-2024_12h50m_every_10-epoch=0069-step=3430.onnx"

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

    data = img.data.copy()
    pt_data = torch.tensor(data,device=device)
    #print(f"pt_data shape: {pt_data.shape}\n")
    ch3_data = bw_1_to_3ch(pt_data,data_format='HW')
    #print(f"ch3_data shape: {ch3_data.shape}\n")
    norm_ch3_data = normalize_in_range(ch3_data,0.0,1.0)
    #print(f"norm_ch3_data shape: {norm_ch3_data.shape}\n")


    if norm_ch3_data.shape[-1] < 864 and norm_ch3_data.shape[-2] < 864:
        pad_flag = True
        mod_data = pad_to_targetM_2d(norm_ch3_data,(864,864),'NCHW')
        print(f"pad_flag (shape): {mod_data.shape}\n")
        print("I'm here")
    elif norm_ch3_data.shape[-1] > 864 or norm_ch3_data.shape[-2] > 864:
        resize_flag = True
        original_shape = (norm_ch3_data.shape[-2],norm_ch3_data.shape[-1])
        mod_data = v2.functional.resize(norm_ch3_data,(864,864),interpolation=v2.InterpolationMode.BICUBIC)
        print(f"resize_flag (shape): {mod_data.shape}\n")
    else:
        mod_data = norm_ch3_data


    #pad_data = pad_to_targetM_2d(norm_ch3_data,(864,864),'NCHW')

    out = mod_data.detach().cpu().numpy().squeeze()

    print("I'm here")

    if debug == True:
        name = f"{img.name}_Pad"
        add_kwargs = {"name":f"{name}"}
        layer_type = "image"
        layer_padded = Layer.create(out,add_kwargs,layer_type)
        layers_out.append(layer_padded)

    if pad_flag:
        offset_0 = out[0].shape[0] - data.shape[0]
        offset_1 = out[0].shape[1] - data.shape[1]
        start_0 = int(offset_0/2)
        start_1 = int(offset_1/2)
        end_0 = int(out[0].shape[0] - start_0)
        end_1 = int(out[0].shape[1] - start_1)

    x = normalize_in_range(mod_data,0,1)
    mean,std = x.mean([0,2,3]),x.std([0,2,3])
    norm = v2.Normalize(mean,std)
    x_norm = norm(x)
    x_norm2 = normalize_in_range(x_norm,0,1)

    #x_eq = equalize_clahe(x_norm2)
    x_eq = equalize_clahe(x_norm2,clip_limit=3.0)

    if log_adjust == True:
        #x_eq = adjust_log(x,gain=1)
        x_eq = adjust_log(x_eq,gain=1)

    if DoG == True:
        diff_of_gauss = DiffOfGausPred(**dog_params)
        x_eq = diff_of_gauss(x_eq)

    if blur == True:
        x_eq = gaussian_blur2d(x_eq,kernel_size=3,sigma=(1.0,1.0),border_type='reflect')
        #x = normalize_in_range(x_eq,0,1)
        x_eq = normalize_in_range(x_eq,0,1)

    #print(f"x shape: {x_eq.shape}\n")

    ENCODER = "efficientnet-b5"
    ENCODER_WEIGHTS = "imagenet"
    CLASSES = [
        "vessel"
    ]
    ACTIVATION = "sigmoid"

    '''
    model = smp.Unet(encoder_name=ENCODER, # smp.UnetPlusPlus(encoder_name=ENCODER,
                    encoder_weights=ENCODER_WEIGHTS,
                    classes=len(CLASSES),
                    activation=ACTIVATION)
    state_dict = torch.load(state_dict_path,map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model_dev = model.to(device)
    output = model_dev.predict(x_eq)
    '''

    #show_info(f'x_eq: {x_eq.shape}')

    x_eq_cpu = x_eq.detach().cpu().numpy()
    pre_poc = x_eq.squeeze().mean(dim=0).detach().cpu().numpy()
    pre_proc_final = pre_poc[start_0:end_0,start_1:end_1]

    if output_preproc == True:
        name = f"{img.name}_Preproc"
        add_kwargs = {"name":f"{name}"}
        layer_type = "image"

        layer = Layer.create(pre_proc_final,add_kwargs,layer_type)
        layers_out.append(layer)

    # start onnx
    onnx_session = InferenceSession(onnx_path)
    input_name = onnx_session.get_inputs()[0].name

    onnx_inputs = {input_name: x_eq_cpu}
    onnx_outs = onnx_session.run(None, onnx_inputs)
    onnx_out = onnx_outs[0].squeeze().astype(np.uint8)
    
    #seg_out = onnx_out.detach().cpu().numpy().squeeze().astype(int)

    if pad_flag:
        final_seg = onnx_out[start_0:end_0,start_1:end_1]
    elif resize_flag:
        final_seg = v2.functional.resize(onnx_out,original_shape,v2.InterpolationMode.NEAREST_EXACT).astype(bool)*1
    else:
        final_seg = onnx_out

    name = f"{img.name}_Seg"
    add_kwargs = {"name":f"{name}"}
    layer_type = "labels"
    layer = Layer.create(final_seg,add_kwargs,layer_type)
    layers_out.append(layer)
    
    # clean up
    del final_seg, onnx_session
    #del seg_out
    #del output
    #del model_dev
    #del model
    del x_eq
    del x_norm2
    del x_norm
    del norm
    del mean
    del std
    del x
    del out
    del pad_data
    del norm_ch3_data
    del ch3_data
    del pt_data

    gc.collect()
    torch.cuda.empty_cache()

   

    return layers_out