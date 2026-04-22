"""
This module contains function code for segmenting images
"""

import gc
import platform
from scipy.ndimage import uniform_filter
import cv2
from typing import Dict, List, Tuple

import numpy as np
from napari.layers import Layer
from napari.types import ImageData, Optional
from napari_cool_tools_io import device, torch
from tqdm import tqdm

from napari_cool_tools_img_proc import DType
from napari_cool_tools_img_proc._normalization_funcs import convert_dtype_and_rescale
from napari_cool_tools_segmentation import (
    BscanSegmentationType,
    EnfaceSegmentationType,
    Path,
    onnx_bscan_melanoma_seg_path,
)  # onnx_bscan, onnx_enface_vessels, onnx_enface_ridge
import onnxruntime as ort
import torch.nn.functional as F

def bscan_yolo_melanoma_seg_func(img_volume: np.ndarray, target_shape:list = [800,800], vmin = 0.3, vmax = 2.0,
                                 CONF_THRESH = 0.45, IOU_THRESH = 0.05, MIN_CONSECUTIVE_FRAMES = 30) -> np.ndarray:
    #TODO function to infere yolo bscan melanoma segmentation model and return labels in same format as bscan_onnx_seg_func for consistency across models and ease of use in napari plugin.
    def build_session(model_path: str) -> ort.InferenceSession:
        """Create an ONNX Runtime session. Prefer CUDA, fallback to CPU."""
        available = ort.get_available_providers()
        print(f"[INFO] Available ORT providers: {available}")

        if "CUDAExecutionProvider" in available:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sess = ort.InferenceSession(
                    model_path,
                    providers=[
                        ("CUDAExecutionProvider", {"device_id": 0}),
                        "CPUExecutionProvider",
                    ],
                )
            active = sess.get_providers()
            if "CUDAExecutionProvider" in active:
                print("[INFO] Running on GPU (CUDAExecutionProvider)")
            else:
                print(
                    "[WARN] CUDAExecutionProvider listed but failed to initialise. "
                    "Falling back to CPU."
                )
        else:
            print("[WARN] CUDA not available using CPU")
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        print(f"[INFO] Active providers: {sess.get_providers()}")
        return sess


    def inspect_model(sess: ort.InferenceSession) -> Tuple[str, Tuple]:
        """Print model I/O details and return (input_name, input_shape)."""
        print("\n--- Model Inputs ---")
        for inp in sess.get_inputs():
            print(f"  name={inp.name!r}  shape={inp.shape}  dtype={inp.type}")

        print("--- Model Outputs ---")
        for out in sess.get_outputs():
            print(f"  name={out.name!r}  shape={out.shape}  dtype={out.type}")
        print()

        inp0 = sess.get_inputs()[0]
        return inp0.name, tuple(inp0.shape)


    
    def preprocess(image_bgr: np.ndarray, input_size: int) -> np.ndarray:
        """
        Resize -> BGR->RGB -> normalize [0,1] -> HWC->CHW -> add batch dim.
        Output shape: (1, 3, H, W)
        """

        #resize image using 
        img = cv2.resize(image_bgr, (input_size, input_size))
        img = img.astype(np.float32)

        #stack up image to 3 channels using numpy
        img = np.stack([img] * 3, axis=-1)
        # print(f"Preprocessed image shape: {img.shape}")

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img)


    def postprocess(
        raw: np.ndarray,
        orig_h: int,
        orig_w: int,
        input_size: int,
        conf_thresh: float,
        iou_thresh: float,
        class_names: List[str],
    ) -> List[Dict]:
        """
        Decode Ultralytics YOLO11 ONNX detection output.

        Expected shape:
        (1, 4 + nc, N)  or  (1, N, 4 + nc)

        Returns a list of dicts with:
        box=(x1,y1,x2,y2), conf, class_id, label
        """
        pred = raw[0]

        # If shape is (N, 4+nc), transpose to (4+nc, N)
        if pred.shape[0] >= pred.shape[1]:
            pred = pred.T

        nc = pred.shape[0] - 4
        actual_nc = max(nc, 1)

        boxes_raw = pred[:4, :]
        scores_raw = pred[4:4 + actual_nc, :]

        class_ids = np.argmax(scores_raw, axis=0)
        confidences = scores_raw[class_ids, np.arange(scores_raw.shape[1])]

        mask = confidences >= conf_thresh
        if not mask.any():
            return []

        boxes_raw = boxes_raw[:, mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        cx, cy, w, h = boxes_raw
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        scale_x = orig_w / input_size
        scale_y = orig_h / input_size

        x1 = (x1 * scale_x).astype(np.float32)
        y1 = (y1 * scale_y).astype(np.float32)
        x2 = (x2 * scale_x).astype(np.float32)
        y2 = (y2 * scale_y).astype(np.float32)

        x1 = np.clip(x1, 0, max(orig_w - 1, 0))
        y1 = np.clip(y1, 0, max(orig_h - 1, 0))
        x2 = np.clip(x2, 0, max(orig_w - 1, 0))
        y2 = np.clip(y2, 0, max(orig_h - 1, 0))

        detections = []
        unique_classes = np.unique(class_ids)

        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = np.stack(
                [x1[cls_mask], y1[cls_mask], x2[cls_mask], y2[cls_mask]],
                axis=1
            )
            cls_confs = confidences[cls_mask]

            nms_boxes = [
                [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
                for b in cls_boxes
            ]

            indices = cv2.dnn.NMSBoxes(
                nms_boxes,
                cls_confs.tolist(),
                conf_thresh,
                iou_thresh
            )

            if indices is None or len(indices) == 0:
                continue

            indices = np.array(indices).reshape(-1)
            label = class_names[cls] if cls < len(class_names) else f"class_{cls}"

            cls_x1 = x1[cls_mask]
            cls_y1 = y1[cls_mask]
            cls_x2 = x2[cls_mask]
            cls_y2 = y2[cls_mask]

            for i in indices:
                detections.append({
                    "box": (
                        float(cls_x1[i]),
                        float(cls_y1[i]),
                        float(cls_x2[i]),
                        float(cls_y2[i]),
                    ),
                    "conf": float(cls_confs[i]),
                    "class_id": int(cls),
                    "label": label,
                })

        detections.sort(key=lambda d: d["conf"], reverse=True)
        return detections
    
    def select_best_detection(detections: List[Dict]) -> Optional[Dict]:
        """Keep only the highest-confidence detection for each frame."""
        if not detections:
            return None
        return max(detections, key=lambda d: d["conf"])

    def filter_continuous_records(
        candidate_records: List[Dict],
        min_consecutive_frames: int = 3,
        keep_only_longest_run: bool = True,
    ) -> List[Dict]:
        """
        Keep only records in continuous frame runs.

        Rules:
        - Frames with no detection are absent from candidate_records already
        - Non-continuous runs are removed
        - By default, keep only the LONGEST continuous run
        - Runs shorter than min_consecutive_frames are removed
        """
        if not candidate_records:
            return []

        runs = split_into_consecutive_runs(candidate_records)
        valid_runs = [run for run in runs if len(run) >= min_consecutive_frames]

        if not valid_runs:
            return []

        if keep_only_longest_run:
            best_run = max(
                valid_runs,
                key=lambda run: (len(run), float(np.mean([x["conf"] for x in run])))
            )
            return sorted(best_run, key=lambda r: r["frame_num"])

        kept = []
        for run in valid_runs:
            kept.extend(run)
        kept.sort(key=lambda r: r["frame_num"])
        return kept
    

    def split_into_consecutive_runs(records: List[Dict]) -> List[List[Dict]]:
        """
        Split records into runs where frame numbers are strictly consecutive:
        ..., 90, 91, 92, ...
        """
        if not records:
            return []

        records = sorted(records, key=lambda r: r["frame_num"])
        runs = [[records[0]]]

        for r in records[1:]:
            prev = runs[-1][-1]
            if r["frame_num"] == prev["frame_num"] + 1:
                runs[-1].append(r)
            else:
                runs.append([r])

        return runs

    ###Main Script###

    # 1) load the yolo onnx file
    onnx_yolo_model_file = onnx_bscan_melanoma_seg_path / "best_cancer.onnx"

    sess = build_session(str(onnx_yolo_model_file))
    input_name, _ = inspect_model(sess)

    # 2) Warm-up
    warm_blob = np.zeros((1, 3, target_shape[0], target_shape[1]), dtype=np.float32)
    _ = sess.run(None, {input_name: warm_blob})


    # prepare input image
    # img_volume = img_volume * 2500 / 65535.0

    #use uniform filter to reduce noise
    img_volume = uniform_filter(img_volume, size=(1, 3, 3))
    # img_volume = averaged_data
    # vmin = 0.3 // 12
    # vmax = 2.0 // 36
    img_volume = np.clip(img_volume, vmin, vmax)
    img_volume = (img_volume - vmin) / (vmax - vmin)
    # img_volume = (adjusted_slice)


    # 4) Run inference over all images
    candidate_records = []
    orig_h, orig_w = img_volume.shape[1:]

    CLASS_NAMES  = ["cancer"]

    for idx, image in enumerate(img_volume):

        blob = preprocess(image, target_shape[0])

        outputs = sess.run(None, {input_name: blob})

        detections = postprocess(
            outputs[0],
            orig_h,
            orig_w,
            target_shape[0],
            CONF_THRESH,
            IOU_THRESH,
            CLASS_NAMES,
        )

        best_det = select_best_detection(detections)

        if best_det is None:
            print(
                f"[{idx}/{len(img_volume)}] no detection"
            )
            continue

        record = {
            "frame_num": idx,
            "orig_box": best_det["box"],
        }
        candidate_records.append(record)

        print(
            f"[{idx}/{len(img_volume)}] conf={best_det['conf']:.3f} "
        )

    # 5) Continuity filtering
    # MIN_CONSECUTIVE_FRAMES = 30
    KEEP_ONLY_LONGEST_RUN  = False

    kept_records = filter_continuous_records(
        candidate_records,
        min_consecutive_frames=MIN_CONSECUTIVE_FRAMES,
        keep_only_longest_run=KEEP_ONLY_LONGEST_RUN,
    )

    # 6) visualizations
    output_label = np.zeros_like(img_volume.data, dtype=int)#generate zero labels for now until function is implemented
    for r in kept_records:
        img = output_label[r["frame_num"]]

        print(f"Frame {r['frame_num']}: box={r['orig_box']}")

        x1, y1, x2, y2 = r["orig_box"] #this corrdinates is normalized to the input size
        x1i, y1i, x2i, y2i = map(lambda v: int(round(v)), (x1, y1, x2, y2))

        img[:,x1i:x2i] = 1

        output_label[r["frame_num"]] = img

    return output_label


def bscan_onnx_seg_func(
    img: ImageData,
    onnx_path=BscanSegmentationType.RETINASEG.value,
    target_shape:list = [864,864], #(992,800)
    batch_size: int = 32,
    num_workers: int = 0,
    gpu_limit: int = 6,
    use_cpu: bool = True,
    output_preproc: bool = False,
    old_preproc: bool = False,
    verbose: bool = True,
    debug: bool = False,
):
    """"""
    import onnxruntime
    import torch.nn as nn
    from jj_nn_framework.data_setup import LoadNumpyData
    from jj_nn_framework.nn_transforms import (
        BscanPreproc2,
        NormalizeCLAHE2,
        PadToTargetM,
        ResizeToFit,
    )
    from torch.utils.data import DataLoader
    from torchvision.transforms.functional import InterpolationMode
    from torchvision.transforms.v2.functional import resize

    if img.dtype.type not in (np.float16,np.float32):
        raise ValueError(f"Image dtype ({img.dtype}) is not np.float16 or np.float32 which is required for this function.\n")
    else:
        if img.dtype.type == np.float16:
            if verbose:
                print("Converting image from np.float16 to np.float32 for segmentation.\n")
            img=convert_dtype_and_rescale(img,datatype=DType.NP_FLOAT32)

    #target_shape = (992, 800)
    init_shape = (img.shape[-2], img.shape[-1])

    if use_cpu:
        processor = "cpu"
        onnx_dev = "cpu"
        if verbose:
            print(f"Using device {platform.processor()}")
    else:
        processor = device
        onnx_dev = "cuda"
        device_id = torch.cuda.current_device()
        if verbose:
            print(f"Using device {torch.cuda.get_device_name(device_id)}\n")

    if verbose:
        print(f"Onnx file_path: {onnx_path}\n")

    num_bscans = len(img)
    rem = num_bscans % batch_size
    if rem != 0:
        missing_bscans = batch_size - rem
        fill_shape = (missing_bscans, img.shape[1], img.shape[2])
        batch_fill = np.empty(fill_shape, dtype=img.dtype)
        img = np.concatenate([img, batch_fill])

    onnx_folder_path = Path(onnx_path).parents[0]

    if verbose:
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

    if old_preproc:
        pred_trans = nn.Sequential(
            PadToTargetM(**pttm_params),
            BscanPreproc2(**bscan_preproc_params),
        )
    else:
        pred_trans = nn.Sequential(
            ResizeToFit(target_shape), PadToTargetM(**pttm_params), NormalizeCLAHE2()
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
                    "trt_engine_cache_enable": True,  # True,
                    "trt_engine_cache_path": onnx_folder_path,
                    "trt_timing_cache_enable": True,  # True,
                    "trt_timing_cache_path": onnx_folder_path,
                    "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)
                    # "user_compute_stream": str(torch.cuda.Stream().cuda_stream),
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
                    "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)
                    # "user_compute_stream": str(torch.cuda.Stream().cuda_stream),
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

    CLASSES = ["vitreous", "retina", "choroid"]  # replace with parameter

    preproc_bscans = []
    label_preds = []

    for image_batch in tqdm(pred_dl, desc="Segmenting B-scans:"):
        # bindtensors to onnx session
        binding = onnx_session.io_binding()

        images_tensor = image_batch.contiguous()
        it_shape = images_tensor.shape

        binding.bind_input(
            name="input",
            device_type=onnx_dev,  #'cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(it_shape),
            buffer_ptr=images_tensor.data_ptr(),
        )

        #pred_shape = (it_shape[0], len(CLASSES), it_shape[2], it_shape[3]) # TODO modify for old model or remove
        pred_shape = (it_shape[0], 1, it_shape[2], it_shape[3])
        pred_tensor = torch.empty(
            #pred_shape, dtype=torch.float32, device=onnx_dev # TODO modify for old model or remove
            pred_shape, dtype=torch.uint8, device=onnx_dev
        ).contiguous()  #'cuda:0').contiguous()
        binding.bind_output(
            name="output",
            device_type=onnx_dev,  #'cuda',
            device_id=0,
            #element_type=np.float32, #TODO modify for old model or remove
            element_type=np.uint8,
            shape=tuple(pred_tensor.shape),
            buffer_ptr=pred_tensor.data_ptr(),
        )

        # run onnx with bidning
        onnx_session.run_with_iobinding(binding)

        # print(f"pred_tensor shape:{pred_tensor.shape}\n")
        # pred_tensor = pred_tensor.reshape(-1,pred_shape[2],pred_shape[3])
        # print(f"pred_tensor shape:{pred_tensor.shape}\n")

        labels = []

        for i, mask in enumerate(pred_tensor):
            # TODO modify for old model or remove
            # label = torch.zeros_like(mask[0], dtype=torch.uint8)
            # mask_argmax = mask.argmax(0)
            # for i, m in enumerate(mask):
            #     label[mask_argmax == i] = i

            labels.append(mask)
            #labels.append(label)

        # print(f"label shape: {labels[0].shape}\n")
        labels = torch.stack(labels, dim=0)

        pred_tensor = pred_tensor.detach().squeeze().cpu().numpy()
        labels = labels.detach().squeeze().cpu().numpy()
        # pred_tensor = labels
        image_batch = image_batch.detach().squeeze().cpu().numpy()

        image_batch = image_batch[:num_bscans]
        # pred_tensor = pred_tensor[:num_bscans]

        # print(f"labels shape: {labels.shape}, pred_tensor shape: {pred_tensor.shape}\n")

        preproc_bscans.append(image_batch)

        # label_preds.append(pred_tensor[:,1,:,:])
        label_preds.append(labels)

    gpu_mem_clear = torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0
    if verbose:
        print(f"GPU memory is clear: {gpu_mem_clear}\n")

    del (
        pred_ds,
        pred_dl,
        # image_batch,
        images_tensor,
        #label, # TODO modify for old model or remove
        #mask_argmax, # TODO modify for old model or remove
        mask,
        #m, # TODO modify for old model or remove
    )
    gc.collect()
    torch.cuda.empty_cache()

    gpu_mem_clear = torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0

    if verbose:
        print(f"GPU memory is clear: {gpu_mem_clear}\n")
        if not gpu_mem_clear:
            print(f"{torch.cuda.memory_summary()}\n")

    preproc_bscans = np.concatenate(
        preproc_bscans, axis=0
    )  # torch.concat(preproc_bscans,dim=0).detach().squeeze().cpu().numpy()
    label_preds = np.concatenate(label_preds, axis=0)
    label_preds_out = label_preds[:num_bscans]
    reshaped_out = resize(
        torch.tensor(label_preds_out.copy()),
        (init_shape),
        interpolation=InterpolationMode.NEAREST_EXACT,
    ).numpy()
    # label_preds = np.stack(label_preds,axis=0)

    output = []

    if output_preproc:
        output.append((preproc_bscans[:num_bscans], "image"))

    # output.append((label_preds[:num_bscans],'labels'))
    output.append((reshaped_out, "labels"))

    return output


def enface_onnx_seg_func(
    data: ImageData,
    onnx_path=EnfaceSegmentationType.VESSEL.value,
    segmentation_type="vessel",
    # segmentation:Literal["optic_nerve","vessel"],
    label_val: int = 1,
    use_cpu: bool = True,
    DoG: bool = False,
    blur: bool = False,
    log_adjust: bool = False,
    output_preproc: bool = False,
    debug: bool = False,
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
    from jj_nn_framework.image_funcs import (
        bw_1_to_3ch,
        normalize_in_range,
        pad_to_targetM_2d,
    )
    from jj_nn_framework.nn_transforms import DiffOfGausPred
    from kornia.enhance import adjust_log, equalize_clahe
    from kornia.filters import gaussian_blur2d
    from napari_cool_tools_io import device
    from onnxruntime import InferenceSession
    from torchvision.transforms import v2

    if use_cpu:
        device = "cpu"

    pad_flag = False
    resize_flag = False

    dog_params = {
        "low_sigma": 0.5,  # 0.0, #1.0,
        "high_sigma": 6.0,  # 20.0,
        "truncate": 4.0,
        "gamma": 1.0,  # 1.2,
        "gain": 1.0,
    }

    data = data.copy()

    if data.dtype == "float64":
        data = data.astype("float32")
    elif data.dtype == "uint8":
        # data = normalize_data_in_range_pt_func()
        data = normalize_in_range(data.astype("float32"), min_val=0.0, max_val=1.0)
    elif data.dtype != "float32":
        raise ValueError(
            f"{data.dtype} is not supported float32, float64, and uint8 are supported"
        )

    pt_data = torch.tensor(data, device=device)
    # print(f"pt_data shape: {pt_data.shape}\n")

    ch3_data = bw_1_to_3ch(pt_data, data_format="HW")
    # print(f"ch3_data shape: {ch3_data.shape}\n")
    norm_ch3_data = normalize_in_range(ch3_data, 0.0, 1.0)
    # print(f"norm_ch3_data shape: {norm_ch3_data.shape}\n")

    if norm_ch3_data.shape[-1] < 864 and norm_ch3_data.shape[-2] < 864:
        pad_flag = True
        mod_data = pad_to_targetM_2d(norm_ch3_data, (864, 864), "NCHW")
        print(f"pad_flag (shape): {mod_data.shape}\n")
    elif norm_ch3_data.shape[-1] > 864 or norm_ch3_data.shape[-2] > 864:
        resize_flag = True
        original_shape = (norm_ch3_data.shape[-2], norm_ch3_data.shape[-1])
        mod_data = v2.functional.resize(
            norm_ch3_data, (864, 864), interpolation=v2.InterpolationMode.BICUBIC
        )
        print(f"resize_flag (shape): {mod_data.shape}\n")
    else:
        mod_data = norm_ch3_data

    # pad_data = pad_to_targetM_2d(norm_ch3_data,(864,864),'NCHW')

    out = mod_data.detach().cpu().numpy().squeeze()

    if pad_flag:
        offset_0 = out[0].shape[0] - data.shape[0]
        offset_1 = out[0].shape[1] - data.shape[1]
        start_0 = int(offset_0 / 2)
        start_1 = int(offset_1 / 2)
        end_0 = int(out[0].shape[0] - start_0)
        end_1 = int(out[0].shape[1] - start_1)

    x = normalize_in_range(mod_data, 0, 1)
    mean, std = x.mean([0, 2, 3]), x.std([0, 2, 3])
    norm = v2.Normalize(mean, std)
    x_norm = norm(x)
    x_norm2 = normalize_in_range(x_norm, 0, 1)

    # x_eq = equalize_clahe(x_norm2)
    x_eq = equalize_clahe(x_norm2, clip_limit=3.0)

    if log_adjust:
        # x_eq = adjust_log(x,gain=1)
        x_eq = adjust_log(x_eq, gain=1)

    if DoG:
        diff_of_gauss = DiffOfGausPred(**dog_params)
        x_eq = diff_of_gauss(x_eq)

    if blur:
        x_eq = gaussian_blur2d(
            x_eq, kernel_size=3, sigma=(1.0, 1.0), border_type="reflect"
        )
        # x = normalize_in_range(x_eq,0,1)
        x_eq = normalize_in_range(x_eq, 0, 1)

    x_eq_cpu = x_eq.detach().cpu().numpy()

    # start onnx
    onnx_session = InferenceSession(onnx_path)
    input_name = onnx_session.get_inputs()[0].name

    # TODO either retrain network with 3 channels or come up with new fix
    # print(f"x_eq_cpu shape {x_eq_cpu.shape}\n")
    # x_eq_cpu = x_eq_cpu[:,0,:,:]
    # print(f"x_eq_cpu shape {x_eq_cpu.shape}\n")
    # x_eq_cpu = x_eq_cpu[:,None,:,:]

    onnx_inputs = {input_name: x_eq_cpu}
    onnx_outs = onnx_session.run(None, onnx_inputs)
    onnx_out = onnx_outs[0].squeeze().astype(np.uint8)

    # seg_out = onnx_out.detach().cpu().numpy().squeeze().astype(int)

    if pad_flag:
        final_seg = onnx_out[start_0:end_0, start_1:end_1].astype(bool) * (label_val)
    elif resize_flag:
        final_seg = v2.functional.resize(
            torch.tensor(onnx_out).unsqueeze(0),
            original_shape,
            v2.InterpolationMode.NEAREST_EXACT,
        ).numpy().astype(bool) * (label_val)
    else:
        final_seg = onnx_out.astype(bool) * (label_val)

    # clean up
    del onnx_session
    # del seg_out
    # del output
    # del model_dev
    # del model
    del x_eq
    del x_norm2
    del x_norm
    del norm
    del mean
    del std
    del x
    del out
    del mod_data
    del norm_ch3_data
    del ch3_data
    del pt_data

    gc.collect()
    torch.cuda.empty_cache()

    return final_seg

def bscan_onnx_deconj_func(
    data: ImageData,
    onnx_path: Path = BscanSegmentationType.DECONJUGATE.value,
    target_bscan_dimension: tuple[int, int] = (512, 1024),
    batch_size: int = 8,  # 32 #16 #8,
    num_workers: int = 0,
    gpu_limit: int = 6,
    use_cpu: bool = False,
    verbose: bool = True,
    debug: bool = False,
) -> tuple[ImageData, str]:
    """"""

    if data.ndim != 3:
        raise ValueError(f"3 dim image is required but {data.ndim} was provided")

    import onnxruntime
    import torch.nn as nn
    from jj_nn_framework.data_setup import LoadNumpyData
    from jj_nn_framework.nn_transforms import (
        Normalize,
        PadToTargetM,
        ResizeToFit,
    )
    from napari_cool_tools_img_proc._normalization_funcs import normalize_data_in_range_func
    from torch.utils.data import DataLoader
    from torchvision.transforms.functional import InterpolationMode
    from torchvision.transforms.v2.functional import resize

    data = data.transpose(-3, -1, -2)  # transpose back to original OCT coordinate system
    # normalize the data
    data = normalize_data_in_range_func(data)

    target_shape = target_bscan_dimension  # (512, 1024)
    init_shape = (data.shape[-2], data.shape[-1])

    if use_cpu:
        processor = "cpu"
        onnx_dev = "cpu"
        if verbose:
            print(f"Using device {platform.processor()}")
    else:
        processor = device
        onnx_dev = "cuda"
        device_id = torch.cuda.current_device()
        if verbose:
            print(f"Using device {torch.cuda.get_device_name(device_id)}\n")
    if verbose:
        print(f"Onnx file_path: {onnx_path}\n")

    num_bscans = len(data)
    rem = num_bscans % batch_size
    if rem != 0:
        missing_bscans = batch_size - rem
        fill_shape = (missing_bscans, data.shape[1], data.shape[2])
        batch_fill = np.empty(fill_shape, dtype=data.dtype)
        data = np.concatenate([data, batch_fill])

    onnx_folder_path = Path(onnx_path).parents[0]

    if verbose:
        print(f"onnx_folder_path: {onnx_folder_path}\n")

    pttm_params = {
        "h": target_shape[-2],
        "w": target_shape[-1],
        "X_data_format": "NHW",
        "y_data_format": "NHW",
        "mode": "constant",
        "value": None,
        "pad_gt": False,
        "device": processor,
    }

    # NormalizeCLAHE2()

    pred_trans = nn.Sequential(
        ResizeToFit(target_shape),
        PadToTargetM(**pttm_params),
        #Normalize(),  # Standardize(),Normalize(),
    )

    pred_ds = LoadNumpyData(
        data,
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
                    "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)
                    #"user_compute_stream": str(torch.cuda.Stream().cuda_stream),
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
                    "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)
                    #"user_compute_stream": str(torch.cuda.Stream().cuda_stream),
                },
            ),
            "CPUExecutionProvider",
        ]

    onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    if debug:
        print(f"input_name = {input_name}")
        print(f"output_name = {output_name}")

    preds = []

    for image_batch in tqdm(pred_dl, desc="Removing complex conjugate from B-scans:"):
        # bindtensors to onnx session
        binding = onnx_session.io_binding()

        images_tensor = image_batch.contiguous()
        it_shape = images_tensor.shape

        if debug:
            print(f"image_tensor shape: {it_shape}")

        binding.bind_input(
            name=input_name,
            device_type=onnx_dev,  #'cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(it_shape),
            buffer_ptr=images_tensor.data_ptr(),
        )

        pred_shape = it_shape
        pred_tensor = torch.empty(
            pred_shape, dtype=torch.float32, device=onnx_dev
        ).contiguous()
        binding.bind_output(
            output_name,
            device_type=onnx_dev,  #'cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(pred_tensor.shape),
            buffer_ptr=pred_tensor.data_ptr(),
        )

        # run onnx with binding
        onnx_session.run_with_iobinding(binding)

        pred_tensor = pred_tensor.detach().squeeze().cpu().numpy()
        pred_tensor = pred_tensor[:num_bscans]
        preds.append(pred_tensor)

    gpu_mem_clear = torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0
    if verbose:
        print(f"GPU memory is clear: {gpu_mem_clear}\n")

    del (
        pred_ds,
        pred_dl,
        images_tensor,
    )
    gc.collect()
    torch.cuda.empty_cache()

    gpu_mem_clear = torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0

    if verbose:
        print(f"GPU memory is clear: {gpu_mem_clear}\n")
        if not gpu_mem_clear:
            print(f"{torch.cuda.memory_summary()}\n")

    preds = np.concatenate(preds, axis=0)
    preds_out = preds[:num_bscans]

    reshaped_out = resize(
        torch.tensor(preds_out.copy()),
        (init_shape),
        interpolation=InterpolationMode.BICUBIC,
    ).numpy()

    reshaped_out = reshaped_out.transpose(
        -3, -1, -2
    )  # transpose back to Napari coordinate system

    output = (reshaped_out, "deconjucated")

    return output