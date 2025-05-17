## File Structure and Descriptions

 Each file serves a different purpose related to OCT or OCTA data processing, visualization, or utility functions. Consult the **How to Run** section for specific usage instructions.

1. **_parse_ini.py**  
   Parses `.ini` configuration files for OCT data (e.g., width, height, frame count). Provides a MagicGUI interface to select a file and process its parameters.

2. **batch_process_profs.py**  
   Allows batch processing of `.prof` files. Includes averaging, segmentation, and enface-generation features. Integrates with napari via `@magic_factory`.

3. **benFastCC_Mod.py / benFastCC.py**  
   Perform fast curve corrections on 3D datasets (e.g., spherical-to-Cartesian transforms) using GPU acceleration. Can generate volumetric and surface renderings.

4. **calc_gpu_batches.py**  
   Simple script to estimate how many B-scans can fit into GPU memory based on the shape of a loaded `.prof` file.

5. **extract_batches_around_indicies.py**  
   Loads a serialized PyTorch dictionary (containing images and labels), extracts specific slices around given indices, then saves them for training or analysis.

6. **generate_bscan_variants.py**  
   Creates multiple B-scan variations (averaged, preprocessed, volume-processed, etc.). Saves the output along with labels in `.pt` format for machine learning tasks.

7. **generate_desktop_octa.py**  
   Demonstrates a specialized flow to produce OCTA data from `.prof` files (calculating std-based variance in blocks) and optionally shows them in napari.

8. **generate_variant_folds.py**  
   Combines multiple training folds into one large dataset. Useful for cross-validation or training pipelines where you merge folds.

9. **oct_unp_to_tiff_pt.py / oct_unp_to_tiff.py**  
   Processes `.unp` files (raw OCT data), applying DC subtraction, Hamming windowing, dispersion correction, Fourier transforms, etc. Outputs processed volumes (tiff, pt, etc.).

10. **og_batch_process_weird_bug.py**  
   An older or experimental batch script illustrating structural and OCTA data handling. Contains MagicGUI-based UI elements.

11. **save_unp_slice.py**  
   Extracts a portion of `.unp` data (e.g., specific frame ranges) and saves them into a new `.unp` file. Useful for partial data debugging or analysis.

12. **susan_data_polars.py / susan_data_polars2.py**  
   Excel/CSV data analysis scripts using the `polars` library. Mostly for reading and filtering clinical data, not strictly OCT processing.

13. **update_meta_files.py**  
   Automates `.ini` file generation or updates the XML metadata for `.unp` or `.prof` files. Can then invoke an external batch processor.

14. **view_bscan_variants.py**  
   Loads a `.pt` file (holding multiple B-scan variants and labels) and displays them in napari for inspection.

15. **view_training_folds.py**  
   Loads combined training folds (images, labels, metadata) into napari. Also shows which fold is marked as a test set.

---

## How to Run Each Script

Below are generic usage tips. If a script has a MagicGUI interface, run it either by calling the file in Python or by opening it within napari.

1. **_parse_ini.py**  
   ```bash
   python batch-tools/_parse_ini.py
   ```
   A small napari GUI will appear. Select your .ini file via the interface to parse OCT metadata.

batch_process_profs.py

```bash
python batch-tools/batch_process_profs.py
```
This script has a @magic_factory interface. After it launches, you can set target_dir for your .prof files, adjust parameters (e.g., averaging, segmentation), and start batch processing.

benFastCC_Mod.py / benFastCC.py

```bash
python batch-tools/benFastCC.py
```
Replace benFastCC.py with benFastCC_Mod.py if desired. Adjust sweep angles, chunk sizes, or resolution inside the file or by editing the magicgui parameters.

calc_gpu_batches.py

```bash
python batch-tools/calc_gpu_batches.py
```
Loads one .prof file to check GPU memory usage for a single B-scan. This helps you guess feasible batch sizes.

extract_batches_around_indicies.py

```bash

python batch-tools/extract_batches_around_indicies.py
```
Provide data (a .pt file) and the output_dir in the GUI. You can specify the slice indices, batch_size, etc. The script will save the new .pt.

generate_bscan_variants.py

```bash
python batch-tools/generate_bscan_variants.py
```
This will open a MagicGUI form allowing you to choose an image volume, label volume, output folder, etc. It then creates multiple B-scan forms and saves them to a .pt file.

generate_desktop_octa.py

```bash
python batch-tools/generate_desktop_octa.py
```
A specialized script for producing an OCTA volume from standard .prof. Configure target_file and save_dir in the UI.

generate_variant_folds.py

```bash
python batch-tools/generate_variant_folds.py
```
Combine multiple .pt fold files located in one directory into a single training dataset.

oct_unp_to_tiff_pt.py / oct_unp_to_tiff.py

```bash
python batch-tools/oct_unp_to_tiff.py
```
You must set the .unp file path in the GUI and then specify transformations (DC subtraction, dispersion correction, etc.). The processed result is shown in napari or saved as tiff/pt.

og_batch_process_weird_bug.py

```bash
python batch-tools/og_batch_process_weird_bug.py
```
Similar to batch_process_profs.py, but for older or experimental usage. Run and set desired toggles (like gen_enface or save_struct) inside the MagicGUI interface.

save_unp_slice.py

```bash
python batch-tools/save_unp_slice.py
```
Manually change unp_file_path, start, and stop in the script’s MagicGUI. It will take that range of frames and write them to a new .unp output file.

susan_data_polars.py / susan_data_polars2.py

```bash
python batch-tools/susan_data_polars.py
```
Adjust the input Excel path (in_path) inside the script as needed. It reads the file using polars and performs data transformations or filters.

update_meta_files.py

```bash
python batch-tools/update_meta_files.py
```
Provide the inputfolder, outputfolder, .ini template, and the global .ini. The script updates metadata and optionally triggers .prof file generation using OCTProcess.exe.

view_bscan_variants.py

```bash
python batch-tools/view_bscan_variants.py
```
Loads a .pt containing multiple image variants and a label set. Napari will open them for inspection.

view_training_folds.py

```bash
python batch-tools/view_training_folds.py
```
Similar to view_bscan_variants.py but specifically for merged folds. A test_fold_path attribute is also displayed.

