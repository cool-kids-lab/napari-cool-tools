## File Structure and Descriptions

### `napari_cool_tools_registration/`

- **`_curve_correction.py`**  
  Implements a 2D curve correction approach for OCT B-scans using transformations from cylindrical or spherical coordinates. Includes a `Curve_Correction_Widget` (Qt-based) for napari integration that handles user input for pivot point, imaging range, scan angle, etc.

- **`_enfaces_registration.py`**  
  Provides functions and classes for registering enface images (2D slices) or related masks. Offers tools such as ellipse detection, center alignment, and polar transformations to help align structures (e.g., optic nerve or vessels in enface images). Also includes a “magic_factory” plugin entry point for enface registration.

- **`_registration_tools_funcs.py`**  
  Contains lower-level “backbone” functions that support registration tasks, including:
  - `a_scan_correction_func2` for applying sinusoidal/linear transform corrections.
  - `a_scan_subpix_registration` for sub-pixel alignments.
  - `adjust_aspect_ratio` to reshape a volume’s aspect ratio using PyTorch transforms.
  - Basic utility enumerations (`AspectRatioPreservationMode`, `ArrayImplementation`) to handle user configuration.

- **`_registration_tools.py`**  
  Serves as the napari plugin code that links the lower-level functions with thread workers. It offers user-facing entry points for:
  - `a_scan_correction`
  - `a_scan_reg_subpix`
  - `adjust_aspect_ratio_plugin`
  - `optical_flow_registration`
  - `m_scan_registration`
