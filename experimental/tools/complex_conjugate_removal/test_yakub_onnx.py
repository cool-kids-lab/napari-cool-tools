import gc
import platform

import napari

# from pathlib import Path
import numpy as np
from napari.layers import Layer
from napari.types import ImageData
from napari_cool_tools_io import device, torch
from napari_cool_tools_segmentation import (
    Path,
)
from tqdm import tqdm

test_volume = Path(
    r"F:\Registration Sample\Test_Data\All_Peripheral_for registration\2024.09.18\08983951\08983951_13_09_18_processed.prof"
)


def bscan_onnx_seg_func(
    img: ImageData,
    onnx_path: Path = Path(
        r"D:\JJ\Development\Yakub_Complex_Conjugate_Processing\U-net_resources\onnx\conjugate_clean.onnx"
    ),  # BscanSegmentationType.BSCAN.value,
    target_bscan_dimension: tuple[int, int] = (512, 1024),
    batch_size: int = 8,  # 32 #16 #8,
    num_workers: int = 0,
    gpu_limit: int = 6,
    use_cpu: bool = True,
    # output_preproc: bool = False,
    old_preproc: bool = False,
    debug: bool = False,
):
    """"""
    import onnxruntime
    import torch.nn as nn
    from jj_nn_framework.data_setup import LoadNumpyData
    from jj_nn_framework.nn_transforms import (
        BscanPreproc2,
        Normalize,
        NormalizeCLAHE2,
        PadToTargetM,
        ResizeToFit,
    )
    from torch.utils.data import DataLoader
    from torchvision.transforms.functional import InterpolationMode
    from torchvision.transforms.v2.functional import resize

    img = img.transpose(-3,-1,-2)

    target_shape = target_bscan_dimension  # (512, 1024)
    init_shape = (img.shape[-2], img.shape[-1])

    if use_cpu:
        processor = "cpu"
        onnx_dev = "cpu"
        print(f"Using device {platform.processor()}")
    else:
        processor = device
        onnx_dev = "cuda"
        device_id = torch.cuda.current_device()
        print(f"Using device {torch.cuda.get_device_name(device_id)}\n")

    print(f"Onnx file_path: {onnx_path}\n")

    num_bscans = len(img)
    rem = num_bscans % batch_size
    if rem != 0:
        missing_bscans = batch_size - rem
        fill_shape = (missing_bscans, img.shape[1], img.shape[2])
        batch_fill = np.empty(fill_shape, dtype=img.dtype)
        img = np.concatenate([img, batch_fill])

    onnx_folder_path = Path(onnx_path).parents[0]

    print(f"onnx_folder_path: {onnx_folder_path}\n")

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

    bscan_preproc_params = {
        "log_gain": 2.5,
        "clahe_clip_limit": 1.0,
        "b_blur_ks": (5, 5),
        "b_blur_sc": 0.1,
        "b_blur_ss": (1.0, 1.0),
        "b_blur_bt": "reflect",
        "g_blur_ks": (5, 5),
        "g_blur_s": (1.0, 1.0),
        "g_blur_bt": "reflect",
    }

    NormalizeCLAHE2()

    if old_preproc:
        pred_trans = nn.Sequential(
            PadToTargetM(**pttm_params),
            BscanPreproc2(**bscan_preproc_params),
        )
    else:
        pred_trans = nn.Sequential(
            ResizeToFit(target_shape),
            PadToTargetM(**pttm_params),
            Normalize(),  # Standardize(),Normalize(),
        )

    pred_ds = LoadNumpyData(
        img,
        # img.transpose(-3,-1,-2), #transpose to work with Yakub's network
        chunk_size=batch_size,
        transform=pred_trans,
        preprocessing=None,
        device=processor,
    )

    pred_dl = DataLoader(
        pred_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    if use_cpu:
        providers = [
            "CPUExecutionProvider",
        ]
    else:
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": device_id,  # Select GPU to execute
                    "trt_max_workspace_size": gpu_limit
                    * 1024
                    * 1024
                    * 1024,  # Set GPU memory usage limit
                    "trt_fp16_enable": True,  # Enable FP16 precision for faster inference
                    # "trt_int8_enable": True, # Enable INT8 precision for quantized inference
                    "trt_engine_cache_enable": True,  # True,
                    "trt_engine_cache_path": onnx_folder_path,
                    "trt_timing_cache_enable": True,  # True,
                    "trt_timing_cache_path": onnx_folder_path,
                    "trt_engine_hw_compatible": False,
                    # "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)
                    "user_compute_stream": str(torch.cuda.Stream().cuda_stream),
                    # "trt_profile_min_shapes": f"input:1x1x{target_shape[-2]}x{target_shape[-1]}",
                    # "trt_profile_opt_shapes": f"input:32x1x{target_shape[-2]}x{target_shape[-1]}",
                    # "trt_profile_max_shapes": f"input:32x1x{target_shape[-2]}x{target_shape[-1]}",
                },
            ),
            (
                "CUDAExecutionProvider",
                {
                    "device_id": device_id,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": gpu_limit * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                    "cudnn_conv_use_max_workspace": "1",
                    # "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)
                    "user_compute_stream": str(torch.cuda.Stream().cuda_stream),
                },
            ),
            "CPUExecutionProvider",
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

    onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    print(f"input_name = {input_name}")
    print(f"output_name = {output_name}")

    # CLASSES = ["vitreous", "retina", "choroid"]  # replace with parameter

    # preproc_bscans = []
    # label_preds = []
    preds = []

    for image_batch in tqdm(pred_dl, desc="Removing complex conjugate from B-scans:"):
        # bindtensors to onnx session
        binding = onnx_session.io_binding()

        images_tensor = image_batch.contiguous()
        it_shape = images_tensor.shape

        if debug:
            print(f"image_tensor shape: {it_shape}")

        binding.bind_input(
            # name="model_input",
            name=input_name,
            device_type=onnx_dev,  #'cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(it_shape),
            buffer_ptr=images_tensor.data_ptr(),
        )

        # pred_shape = (it_shape[0], len(CLASSES), it_shape[2], it_shape[3])
        pred_shape = it_shape
        pred_tensor = torch.empty(
            pred_shape, dtype=torch.float32, device=onnx_dev
        ).contiguous()  #'cuda:0').contiguous()
        binding.bind_output(
            # name="output",
            output_name,
            device_type=onnx_dev,  #'cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(pred_tensor.shape),
            buffer_ptr=pred_tensor.data_ptr(),
        )

        # run onnx with bidning
        onnx_session.run_with_iobinding(binding)

        # print(f"pred_tensor shape:{pred_tensor.shape}\n")
        # pred_tensor = pred_tensor.reshape(-1,pred_shape[2],pred_shape[3])
        # print(f"pred_tensor shape:{pred_tensor.shape}\n")

        # labels = []

        # for i, mask in enumerate(pred_tensor):
        #     label = torch.zeros_like(mask[0], dtype=torch.uint8)
        #     mask_argmax = mask.argmax(0)
        #     for i, m in enumerate(mask):
        #         label[mask_argmax == i] = i

        #     labels.append(label)

        # # print(f"label shape: {labels[0].shape}\n")
        # labels = torch.stack(labels, dim=0)

        pred_tensor = pred_tensor.detach().squeeze().cpu().numpy()
        # labels = labels.detach().squeeze().cpu().numpy()
        # pred_tensor = labels
        # image_batch = image_batch.detach().squeeze().cpu().numpy()

        # image_batch = image_batch[:num_bscans]
        pred_tensor = pred_tensor[:num_bscans]

        # print(f"labels shape: {labels.shape}, pred_tensor shape: {pred_tensor.shape}\n")

        # preproc_bscans.append(image_batch)

        # label_preds.append(pred_tensor[:,1,:,:])
        # label_preds.append(labels)
        preds.append(pred_tensor)

    gpu_mem_clear = torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0
    print(f"GPU memory is clear: {gpu_mem_clear}\n")

    del (
        pred_ds,
        pred_dl,
        # image_batch,
        images_tensor,
        # label,
        # mask_argmax,
        # mask,
        # m,
    )
    gc.collect()
    torch.cuda.empty_cache()

    gpu_mem_clear = torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0

    print(f"GPU memory is clear: {gpu_mem_clear}\n")
    if not gpu_mem_clear:
        print(f"{torch.cuda.memory_summary()}\n")

    # preproc_bscans = np.concatenate(
    #     preproc_bscans, axis=0
    # )  # torch.concat(preproc_bscans,dim=0).detach().squeeze().cpu().numpy()
    # label_preds = np.concatenate(label_preds, axis=0)
    # label_preds_out = label_preds[:num_bscans]
    preds = np.concatenate(preds, axis=0)
    preds_out = preds[:num_bscans]

    # preds_out = preds_out.squeeze().transpose((-3,-1,-2)) # transpose from Yakub's network to Napari orientation

    reshaped_out = resize(
        # torch.tensor(label_preds_out.copy()),
        torch.tensor(preds_out.copy()),
        (init_shape),
        # interpolation=InterpolationMode.NEAREST_EXACT,
        interpolation=InterpolationMode.BICUBIC,
    ).numpy()
    # label_preds = np.stack(label_preds,axis=0)

    reshaped_out = reshaped_out.transpose(-3,-1,-2)
    output = []

    # if output_preproc:
    #     output.append((preproc_bscans[:num_bscans], "image"))

    # output.append((label_preds[:num_bscans],'labels'))
    # output.append((reshaped_out, "labels"))
    output.append((reshaped_out, "image"))

    return output


viewer = napari.Viewer(show=False)
viewer.open(test_volume, plugin="napari-cool-tools-io")
#data = viewer.layers[-1].data.transpose(-3, -1, -2)
data = viewer.layers[-1].data
name = viewer.layers[-1].name
add_kwargs = {"name": f"{name}_deconjugated"}
cleaned, layer_type = bscan_onnx_seg_func(data, batch_size=8, use_cpu=False)[0]
#layer = Layer.create(cleaned.transpose(-3, -1, -2), add_kwargs, layer_type)
layer = Layer.create(cleaned, add_kwargs, layer_type)
viewer.add_layer(layer)
viewer.show()
napari.run()
