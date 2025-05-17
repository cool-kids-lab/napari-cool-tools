# README

## File Structure and Descriptions

### `napari_cool_tools_io/`

- **`_float64_writer.py`**  
  Demonstrates how to save layer data to disk, converting float64 data into 8-bit images for writing via `imageio`.

- **`_import_export_labels.py`**  
  Contains functions to convert napari Labels into a grayscale TIFF-compatible image and vice versa. For example, you can export labeled segmentation data into a grayscale format or import a grayscale-coded TIFF back into label format.

- **`_import_unp.py`**  
  Wraps a small DLL call (`UNPBatchProcessing.dll`) to convert `.unp` files into `.prof` format (Windows-specific). This is a specialized function requiring external DLL usage.

- **`_load_slo.py`**  
  An example function (`load_slo`) for reading `.slo` files (enface data) without standard metadata. Allows some transformations like alignment, subpixel registration, scaling, etc.

- **`_mat_reader.py` / `_mat_writer.py`**  
  Reading and writing `.mat` files via `scipy.io.loadmat` / `savemat`. Supports reading of special napari-style `.mat` scenes or simple data arrays, plus writing compressed/uncompressed scene data.

- **`_prof_reader.py` / `_prof_writer.py`**  
  Code to read and write the COOL Lab’s `.prof` format, including reading meta-information from `.xml` or `.ini` files. While reading, it transposes and flips data for typical napari-friendly orientation. The writer tries to create minimal `.xml` metadata if absent.

- **`_screen_capture.py`**  
  Provides GUI buttons to “capture” the currently viewed slice from napari. This can produce a 2D image layer or raw data slice from a 3D volume.

- **`_torch_reader.py` / `_torch_writer.py`**  
  Reading and writing `.pt` files (PyTorch format). For example, `torch_file_writer` can transpose/flip data to keep a consistent orientation with `.prof` reading, while `torch_file_importer` can handle both raw Tensor or dict structures. Also includes a magic_factory-based widget (`torch_file_exporter`) to export layers in various ways.

- **`_unp_reader.py`**  
  Another specialized COOL Lab file format (`.unp`). Similar to `_prof_reader.py`, it uses an external DLL (`UNPImporter.dll`) for partial reading. The data is then rearranged for napari.


