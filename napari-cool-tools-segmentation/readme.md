## File Structure and Descriptions

### napari_cool_tools_segmentation/

#### `_segmentation_funcs.py`
- **Purpose**: Core Python functions for image segmentation, used by napari plugin code. 
- **Key Functions**:
  - `bscan_onnx_seg_func(...)`: Inference with a B-scan model in ONNX format, handles batching and transforms.
  - `b_scan_pix2pixHD_seg_func(...)`: Inference with a `pix2pixHD`-style model for B-scans.
  - `enface_unet_seg_func(...)`: Basic PyTorch segmentation with a U-Net variant for enface images.
  - `enface_onnx_seg_func(...)`: ONNX-based enface segmentation pipeline (pre-processing, DoG, etc.).

#### `_segmentation.py`
- **Purpose**: Napari plugin interface code, hooking up the core segmentation functions with MagicGui/thread_worker decorators.  
- **Key Decorators**:
  - `@magic_factory` → e.g. `bscan_onnx_seg_plugin(...)`, wraps the `bscan_onnx_seg_func` for user interaction in napari.
  - `@thread_worker` → e.g. `bscan_onnx_seg_thread(...)`, runs segmentation off the main UI thread.
- **Available plugin commands**:
  1. **pix2pix_inference_bscan** → `b_scan_pix2pixHD_seg`
  2. **unet_inference_enface** → `enface_unet_seg`
  3. **onnx_bscan_seg** → `bscan_onnx_seg_plugin`
  4. **onnx_enface_seg** → `enface_onnx_seg_plugin`
  5. **popcorn_enface_seg** → `enface_popcorn_seg_func`

**Each plugin** typically:
- Reads input image from a napari layer
- Pre-processes (padding, normalization, maybe DoG or log transform)
- Runs inference (PyTorch or ONNX)
- Returns new napari layer(s) for preprocessed or label results.
