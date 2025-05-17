## File Structure and Descriptions


### `napari_cool_tools_img_proc/`

- **`_denoise_funcs.py`**  
  Contains functions for denoising operations. Implements a Difference of Gaussians (DoG) filter using both `skimage` and a PyTorch-based approach, and a total variation (TV) denoising routine.

- **`_denoise.py`**  
  Plugin-facing code that sets up napari integration. Includes worker-thread functions to run DoG and TV denoising in the background and add results as new layers.

- **`_equalization_funcs.py`**  
  Implements various equalization operations (e.g., CLAHE, brightness adjustments, background removal). Also includes helper methods like `auto_brightness_adjust` or `init_bscan_preproc` for additional preprocessing steps.

- **`_equalization.py`**  
  Napari plugin code for equalization tasks. Wraps the functions from `_equalization_funcs.py` into thread-safe routines that produce new layers in napari (CLAHE, background removal, histogram matching, etc.).

- **`_filters.py`**  
  Provides GPU-accelerated filtering operations via Kornia, such as bilateral blur, median blur, unsharp masking, and Gaussian blur. Contains the main plugin logic (e.g., `filter_bilateral`, `sharpen_um`, `filter_median`).

- **`_luminance_funcs.py`**  
  Low-level functions that implement log and inverse-log adjustments, optionally using `kornia.enhance` for GPU acceleration. These can be invoked from napari to modify image brightness or contrast.

- **`_luminance.py`**  
  Napari plugin for luminance (brightness/contrast) operations. Wraps and schedules `luminance_funcs.py` routines (log adjustment, gamma correction, etc.) as background jobs.

- **`_morphology_funcs.py`**  
  Defines morphological functions (dilation, erosion) using Kornia's GPU-based morphology. The logic for volumetric or 2D operations is here, controlling kernels and iterations.

- **`_morphology.py`**  
  Plugin code that calls the above morphology functions in worker threads, generating new layers for the results of dilation or erosion.

- **`_nn_tools_2D.py`**  
  Contains helper functions for 2D deep learning data preparation, such as padding or pooling. These are exposed to napari so that users can quickly pad or pool 2D images.

- **`_normalization_funcs.py`**  
  Holds utility functions for standardizing or normalizing image intensity values (range-mapping, standard score). Offers both NumPy and PyTorch-based routines.

- **`_normalization.py`**  
  Napari plugin bridging the `_normalization_funcs.py` calls. Provides thread-safe methods to standardize images, normalize to specific ranges, or do so via GPU-accelerated approaches.

## How to Run Each Script

Find them in napari menu.
