## File Structure and Descriptions

### `_averaging_tools_funcs.py`
**Purpose:** Contains core Python functions for averaging slices in a volumetric dataset (e.g., B-scan data).

**Key Functions:**
- **`average_bscans(vol, scans_per_avg=3)`**  
  Uses a block-reduce approach to average every `scans_per_avg` slices. Returns the averaged volume.

- **`average_per_bscan(vol, scans_per_avg=3, axis=0, trim=False)`**  
  Averages every `scans_per_avg` images, centered on each slice, effectively a sliding-window approach along an axis. Returns an output volume that includes local means for each slice.

- **`average_per_bscan_pt(vol, scans_per_avg=3, axis=0, trim=False, ensemble=True, gauss=False)`**  
  PyTorch-based approach to per-B-scan averaging, optionally performing an “ensemble” average across multiple axes.

---

### `_averaging_tools.py`
**Purpose:** Napari plugin code linking `_averaging_tools_funcs.py` to the UI.

**Primary Functions:**
- **`average_bscans_plugin` / `average_bscans_thread`**  
  Gathers user parameters (`scans_per_avg`), calls `average_bscans(...)`, and returns a new layer in napari.

- **`average_per_bscan_plugin` / `average_per_bscan_thread`**  
  Bridges user input to call either `average_per_bscan_pt(...)` or `average_per_bscan(...)` based on the chosen implementation. Creates a new layer with the averaged volume.

---

### `_masking_tools_funcs.py`
**Purpose:** Provides low-level masking and labeling manipulations for volumetric data, including B-scan label cleanup or relative masking.

**Key Functions:**
- **`circle_circumference_mask(center_x, center_y, radius, image_size)`**  
  Returns a 2D boolean mask representing the circumference of a circle.

- **`bscan_label_cleanup(data, input_label_vals=[0,1,2], ...)`**  
  Uses connected components (e.g., via `cv2`) to identify small or large features in B-scan label volumes and optionally removes or highlights them.

- **`mask_relative_to_existing_label(data, occurence="first", relative="before", axis=0, …)`**  
  Creates a mask based on the position of a reference label along a specified axis (e.g., “before the first occurrence of label X”).

- **`mask_interface_of_existing_label(data, occurence="first", …)`**  
  Generates a mask that outlines the boundary of a label or the region near it.

- **`create_blank_lable_from_layer(img_data)`**  
  Returns an empty `uint8` array with the same shape as the input image data.

- **`isolate_labeled_volume(img_data, lbl_data, label)`**  
  Returns a copy of `img_data` with all elements not matching the specified label set to zero.

- **`project_2d_mask(img_data, lbl_data, axis=1, …)`**  
  Projects a 2D label along a specified axis to create a 3D volume label.

---

### `_masking_tools.py`
**Purpose:** Contains napari plugin layer definitions for the masking tools, integrating the logic from `_masking_tools_funcs.py` into the user interface.

**Primary Plugin Functions:**
- **`bscan_label_cleanup_plugin`**  
  Wraps the cleanup function (`bscan_label_cleanup_thread`) to clean small or large features in B-scan label volumes.

- **`group_labels_plugin`**  
  Merges multiple label values into a single label using a grouping function (e.g., `group_labels(...)`).

- **`mask_relative_to_existing_label_plugin` / `mask_interface_of_existing_label_plugin`**  
  Generates new label layers that show relative positioning or boundary-based masks.

- **`create_blank_lable_from_layer_plugin`**  
  Creates an empty label layer matching the dimensions of an image.

- **`isolate_labeled_volume_plugin`**  
  Calls a thread-enabled version (`isolate_labeled_volume_thread`) to retain only the specified label in the volume.

- **`project_2d_mask_plugin`**  
  Utilizes `project_2d_mask_thread` to replicate a 2D mask across a dimension, producing a new layer.

---

### `_measuring_tools.py`
**Purpose:** Contains tools for measuring volumes (e.g., calculating the volume of label masks) and for interactive geometry.

**Key Functions:**
- **`calc_label_volumes(layer)`**  
  Performs a simple count of non-zero voxels for each label (e.g., differentiating retina versus choroid).

- **`draw_circle_mask`**, **`click_drag`**  
  Provide interactive methods for drawing circle masks or for user-driven measurement.

- **`calc_zone1`**, **`mark_fovea`**, **`mark_disc`**  
  Illustrate advanced interactive geometry functions for specialized workflows, such as measuring the disc-fovea zone in retina images.

---

### `_projection_tools.py`
**Purpose:** Implements code for generating orthogonal maximum intensity projections (MIPs) from volumetric data.

**Primary Functions:**
- **`mip(img, yx=True, zy=False, xz=False)`**  
  Evaluates user-specified flags and calls `mip_thread` to generate the desired projections.

- **`mip_thread`**  
  Performs the actual calculation of MIPs along the selected axes, yielding new napari layers.

---

### `_slicing_shaping_tools.py`
**Purpose:** Facilitates slicing and reshaping volumetric data, including splitting volumes into sub-stacks, reshaping arrays, or stacking multiple selected layers.

**Key Functions:**
- **`reshape_vol`** / **`reshape_vol_thread`**  
  Allows users to specify a new shape (e.g., `(-1, 3, :,:)`) and then applies `numpy.reshape` to modify the volume’s dimensional ordering.

- **`split_vol`**  
  Splits a volume into multiple sub-volumes along a chosen axis. Each resulting sub-volume is added as a new napari layer.

- **`stack_selected`** / **`stack_selected_2D`**  
  Stacks all selected napari layers (sorted by name) along a specified axis to produce a single 3D or 2D dataset.