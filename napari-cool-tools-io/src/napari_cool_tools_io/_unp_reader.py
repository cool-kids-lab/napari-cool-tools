import os.path as ospath
import xml.etree.ElementTree as ET
from pathlib import Path
import configparser
from napari_cool_tools_vol_proc._averaging_tools_funcs import average_bscans_torch
import numpy as np
from napari.utils.notifications import show_info
from napari_cool_tools_io.process_unp import process_unp, process_unp_sine_pause
from qtpy.QtWidgets import QDialog
from napari_cool_tools_io import unp_meta
from napari_cool_tools_io._unp_preview_widget import Unp_Preview_Widget
from napari.layers import Layer
from napari_cool_tools_oct_preproc._oct_preproc_func import generate_octa, OCTACalc

def unp_get_reader(path):
    """Return a reader callable for .unp files, or None if the path is unsupported.

    This function determines whether this plugin can read the given path. If the
    path is a string ending with the ".unp" extension, it returns the corresponding
    reader function that can load the file; otherwise, it returns None so that
    other plugins may attempt to read the data.

    Args:
        path: A candidate path to a file.

    Returns:
        A callable that accepts the path and returns layer data when the path ends
        with ".unp"; otherwise, None.

    Notes:
        - The extension check is case-sensitive (".unp" only).
        - This function does not verify file existence or readability; it performs
          only a lightweight extension check.

    Examples:
        >>> reader = unp_get_reader("images/sample.unp")
        >>> if reader is not None:
        ...     layer_data = reader("images/sample.unp")
    """
    if isinstance(path, str) and path.endswith(".unp"):
        return unp_file_reader
    
    return None

def unp_proc_meta(path) -> unp_meta | None:
    """
    Extract metadata for a .unp file by locating and parsing associated .ini or .xml files.
    Parameters
    ----------
    path : str or os.PathLike
        Path to the .unp file. The function will look for metadata files with the same base
        name and either a .ini or .xml extension in the same directory.
    Returns
    -------
    tuple[int | None, int | None, int | None, int | None, int | None, bool | None, bool | None, str | None] | None
        If metadata is found, returns an 8-tuple:
            (width, height, depth, bmscan, vista, packed, double_side, pattern)
        - width, height, depth, bmscan, vista are returned as ints when present.
        - packed and double_side are returned as bools when present.
        - pattern is returned as a str when present.
        Any field not available in the metadata will be None. If no metadata file (.ini or .xml)
        is present, the function returns None.
    Behavior
    --------
    - Logs progress using show_info().
    - Looks for a .ini file first. If present, reads values via configparser:
        * 'OCTViewer': WIDTH -> width, HEIGHT -> height, FRAMES -> depth
        * 'OCTA': BMScan -> bmscan
        * 'Scanning': VISTA_Num -> vista, Bidirectional -> double_side, Pattern -> pattern
        * 'Acquisition': PACKED12 -> packed
      Values are converted to int or bool as appropriate. If the .ini is successfully read,
      its parsed values are returned.
    - If no .ini file is found but an .xml file exists, parses the XML and extracts:
        * Volume_Size attributes: Height -> height, Width -> width, Number_of_Frames -> depth
        * Scanning_Parameters attribute: Number_of_BM_scans -> bmscan
        * reference arm motor position offset: Motor_pos -> motor_position
      When only XML is used, vista, packed, double_side and pattern remain None.
    - If both files exist, the .ini file takes precedence (checked first).
    - If neither file exists, returns None.
    Exceptions
    ----------
    - configparser.Error (e.g., NoSectionError, NoOptionError) or ValueError may be raised when reading
      or converting INI values.
    - xml.etree.ElementTree.ParseError may be raised for malformed XML.
    - OSError/IOError may be raised for underlying file access issues.
    Examples
    --------
    >>> unp_proc_meta('/path/to/scan.unp')
    (4096, 800, 840, 2, 1, True, False, 'Raster')
    >>> unp_proc_meta('/path/to/scan_without_meta.unp')
    """
    show_info(f"\nOpening file: {path}")

    head, tail = ospath.split(path)
    file_no_ext = tail.split(".")[0]

    # constuct path to metafile assumed to be in same directory
    meta_path_xml = ospath.join(head, file_no_ext + ".xml")
    show_info(f"Associated .xml meta data file: {meta_path_xml}")

    meta_path_ini = ospath.join(head, file_no_ext + ".ini")
    show_info(f"Associated .ini meta data file: {meta_path_ini}")

    # Initialize metadata container
    meta = unp_meta()
    #width, height, depth = [4096, 800, 840]

    if Path(meta_path_ini).is_file():
        show_info(".ini Meta Data exists:")

        config = configparser.ConfigParser()
        config.read(meta_path_ini)

        meta.width = config.getint('General', 'WIDTH')
        meta.height = config.getint('General', 'HEIGHT')
        meta.depth = config.getint('General', 'FRAMES')
        meta.bmscan = config.getint('OCTA', 'BMScan')
        # TODO acquire motor_position from .ini file
        # meta.motor_position = config.getint('Motor_Control', 'Current') confirm with Yakub
        meta.vista = config.getint('Scanning', 'VISTA_Num')
        meta.packed = config.getboolean('Acquisition', 'PACKED12')
        meta.double_side = config.getboolean('Scanning', 'Bidirectional')
        meta.pattern = config['Scanning']['Pattern']
        meta.delay = config.getint('Scanning', 'XDelay')

        if meta.pattern == "Sine_Pause":

            # if config.has_option('Scanning', 'Sine_Pause_Frame_Index'):
            meta.sine_frame_indices = list(map(int, config['Scanning']['Sine_Pause_Frame_Index'].split()))
            meta.sine_hires_ratio = config.getint('Scanning', 'Sine_Pause_X_Rate_Reduction')
            # else:
            #     meta.sine_frame_indices = [236, 256, 286, 306, 336, 356, 386, 406, 434, 454]
            #     meta.sine_hires_ratio = 3

        dialog = Unp_Preview_Widget()
        dialog.set_unp_path(Path(path), meta)

        dialog.doubleSideCheckBox.setChecked(meta.double_side)

        result = dialog.exec_()

        if result == QDialog.Accepted:
            meta.double_side = dialog.doubleSideCheckBox.isChecked()
            meta.full_range = dialog.fullRangeCheckBox.isChecked()
            meta.desine = dialog.desineCheckBox.isChecked()
            meta.dcSubtract = dialog.dcSubtractCheckBox.isChecked()
            meta.log_scale = dialog.logScaleCheckBox.isChecked()
            meta.max_projection = dialog.maxProjectionCheckBox.isChecked()
            meta.c2A = dialog.dispC2ASpinBox.value()
            meta.c3A = dialog.dispC3ASpinBox.value()
            meta.c2B = dialog.dispC2BSpinBox.value()
            meta.c3B = dialog.dispC3BSpinBox.value()
            meta.split_dispersion = dialog.splitDispersionCheckBox.isChecked()
            meta.dispersion_mode = dialog.dispersionModeComboBox.currentIndex()
            
            if dialog.OCTACheckBox.isChecked():
                meta.octa = dialog.OCTAComboBox.currentText()
                meta.structure = dialog.structureCheckBox.isChecked()
            
            meta.windowType = dialog.windowComboBox.currentIndex()
            meta.split_spectrum = dialog.splitSpectrumCheckBox.isChecked()

            print("File Info")
            print(f"width: {meta.width}")
            print(f"height: {meta.height}")
            print(f"depth: {meta.depth}")
            print(f"bmscan: {meta.bmscan}")
            print(f"motor position: {meta.motor_position}")
            print(f"vista: {meta.vista}")
            print(f"packed: {meta.packed}")
            print(f"double_side: {meta.double_side}")
            print(f"pattern: {meta.pattern}")
            print(f"delay: {meta.delay}")
            print(f"full_range: {meta.full_range}")
            print(f"desine: {meta.desine}")
            print(f"dcSubtract: {meta.dcSubtract}")
            print(f"log_scale: {meta.log_scale}")
            print(f"max_projection: {meta.max_projection}")
            print(f"c2A: {meta.c2A}")
            print(f"c3A: {meta.c3A}")
            print(f"c2B: {meta.c2B}")
            print(f"c3B: {meta.c3B}")
            print(f"split_dispersion: {meta.split_dispersion}")
            print(f"octa: {meta.octa}")
            print(f"windowType: {meta.windowType}")
            print(f"split_spectrum: {meta.split_spectrum}")

            return meta
        else:
            return None

    if Path(meta_path_xml).is_file():
        show_info(".xml Meta Data exists:")

        tree = ET.parse(meta_path_xml)
        root = tree.getroot()
        volume_size = root.find(".//Volume_Size")
        volume_size_attrib = volume_size.attrib # type: ignore
        meta.height = int(volume_size_attrib["Height"])
        meta.width = int(volume_size_attrib["Width"])
        meta.depth = int(volume_size_attrib["Number_of_Frames"])

        scanning_params = root.find(".//Scanning_Parameters")
        scanning_params_attrib = scanning_params.attrib # type: ignore
        meta.bmscan = int(scanning_params_attrib["Number_of_BM_scans"])
        motor_pos = scanning_params_attrib.get("Motor_Pos")
        if motor_pos is not None:
            meta.motor_position = int(motor_pos)

        dialog = Unp_Preview_Widget()
        dialog.set_unp_path(Path(path), meta)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            meta.double_side = dialog.doubleSideCheckBox.isChecked()
            meta.full_range = dialog.fullRangeCheckBox.isChecked()
            meta.desine = dialog.desineCheckBox.isChecked()
            meta.dcSubtract = dialog.dcSubtractCheckBox.isChecked()
            meta.log_scale = dialog.logScaleCheckBox.isChecked()
            meta.max_projection = dialog.maxProjectionCheckBox.isChecked()
            meta.c2A = dialog.dispC2ASpinBox.value()
            meta.c3A = dialog.dispC3ASpinBox.value()
            meta.c2B = dialog.dispC2BSpinBox.value()
            meta.c3B = dialog.dispC3BSpinBox.value()
            meta.split_dispersion = dialog.splitDispersionCheckBox.isChecked()
            meta.dispersion_mode = dialog.dispersionModeComboBox.currentIndex()
            #packed is always false for xml only case
            meta.packed = False

            if dialog.OCTACheckBox.isChecked():
                meta.octa = dialog.OCTAComboBox.currentText()
                meta.structure = dialog.structureCheckBox.isChecked()
            
            meta.windowType = dialog.windowComboBox.currentIndex()
            meta.split_spectrum = dialog.splitSpectrumCheckBox.isChecked()

            print("File Info")
            print(f"width: {meta.width}")
            print(f"height: {meta.height}")
            print(f"depth: {meta.depth}")
            print(f"bmscan: {meta.bmscan}")
            print(f"motor position: {meta.motor_position}")
            print(f"vista: {meta.vista}")
            print(f"packed: {meta.packed}")
            print(f"double_side: {meta.double_side}")
            print(f"pattern: {meta.pattern}")
            print(f"delay: {meta.delay}")
            print(f"full_range: {meta.full_range}")
            print(f"desine: {meta.desine}")
            print(f"dcSubtract: {meta.dcSubtract}")
            print(f"log_scale: {meta.log_scale}")
            print(f"max_projection: {meta.max_projection}")
            print(f"c2A: {meta.c2A}")
            print(f"c3A: {meta.c3A}")
            print(f"c2B: {meta.c2B}")
            print(f"c3B: {meta.c3B}")
            print(f"split_dispersion: {meta.split_dispersion}")
            print(f"octa: {meta.octa}")
            print(f"windowType: {meta.windowType}")
            print(f"split_spectrum: {meta.split_spectrum}")

            return meta

        else:
            return None

    # case no metadata request path to metadata or cancel file load
    else:
        return None
    
def unp_batch_proc_meta(path) -> unp_meta | None:
    """
    Extract metadata for a .unp file by locating and parsing associated .ini or .xml files.
    Parameters
    ----------
    path : str or os.PathLike
        Path to the .unp file. The function will look for metadata files with the same base
        name and either a .ini or .xml extension in the same directory.
    Returns
    -------
    tuple[int | None, int | None, int | None, int | None, int | None, bool | None, bool | None, str | None] | None
        If metadata is found, returns an 8-tuple:
            (width, height, depth, bmscan, vista, packed, double_side, pattern)
        - width, height, depth, bmscan, vista are returned as ints when present.
        - packed and double_side are returned as bools when present.
        - pattern is returned as a str when present.
        Any field not available in the metadata will be None. If no metadata file (.ini or .xml)
        is present, the function returns None.
    Behavior
    --------
    - Logs progress using show_info().
    - Looks for a .ini file first. If present, reads values via configparser:
        * 'OCTViewer': WIDTH -> width, HEIGHT -> height, FRAMES -> depth
        * 'OCTA': BMScan -> bmscan
        * 'Scanning': VISTA_Num -> vista, Bidirectional -> double_side, Pattern -> pattern
        * 'Acquisition': PACKED12 -> packed
      Values are converted to int or bool as appropriate. If the .ini is successfully read,
      its parsed values are returned.
    - If no .ini file is found but an .xml file exists, parses the XML and extracts:
        * Volume_Size attributes: Height -> height, Width -> width, Number_of_Frames -> depth
        * Scanning_Parameters attribute: Number_of_BM_scans -> bmscan
      When only XML is used, vista, packed, double_side and pattern remain None.
    - If both files exist, the .ini file takes precedence (checked first).
    - If neither file exists, returns None.
    Exceptions
    ----------
    - configparser.Error (e.g., NoSectionError, NoOptionError) or ValueError may be raised when reading
      or converting INI values.
    - xml.etree.ElementTree.ParseError may be raised for malformed XML.
    - OSError/IOError may be raised for underlying file access issues.
    Examples
    --------
    >>> unp_proc_meta('/path/to/scan.unp')
    (4096, 800, 840, 2, 1, True, False, 'Raster')
    >>> unp_proc_meta('/path/to/scan_without_meta.unp')
    """
    show_info(f"\nOpening file: {path}")

    head, tail = ospath.split(path)
    file_no_ext = tail.split(".")[0]

    # constuct path to metafile assumed to be in same directory
    meta_path_xml = ospath.join(head, file_no_ext + ".xml")
    show_info(f"Associated .xml meta data file: {meta_path_xml}")

    meta_path_ini = ospath.join(head, file_no_ext + ".ini")
    show_info(f"Associated .ini meta data file: {meta_path_ini}")

    # Initialize metadata container
    meta = unp_meta()
    #width, height, depth = [4096, 800, 840]


    # Unp_Preview_Widget()

    if Path(meta_path_ini).is_file():
        show_info(".ini Meta Data exists:")

        config = configparser.ConfigParser()
        config.read(meta_path_ini)

        meta.width = config.getint('General', 'WIDTH')
        meta.height = config.getint('General', 'HEIGHT')
        meta.depth = config.getint('General', 'FRAMES')
        meta.bmscan = config.getint('OCTA', 'BMScan')
        # TODO acquire motor_position from .ini file
        # meta.motor_position = config.getint('Motor_Control', 'Current') confirm with Yakub
        meta.vista = config.getint('Scanning', 'VISTA_Num')
        meta.packed = config.getboolean('Acquisition', 'PACKED12')
        meta.double_side = config.getboolean('Scanning', 'Bidirectional')
        meta.pattern = config['Scanning']['Pattern']
        meta.delay = config.getint('Scanning', 'XDelay')

        if meta.pattern == "Sine_Pause":
            meta.sine_frame_indices = list(map(int, config['Scanning']['Sine_Pause_Frame_Index'].split()))
            meta.sine_hires_ratio = config.getint('Scanning', 'Sine_Pause_X_Rate_Reduction')

        return meta

    elif Path(meta_path_xml).is_file():
        show_info(".xml Meta Data exists:")

        tree = ET.parse(meta_path_xml) # TODO here and above add handling for corrupt or empty xml file
        root = tree.getroot()
        volume_size = root.find(".//Volume_Size")
        volume_size_attrib = volume_size.attrib # type: ignore
        meta.height = int(volume_size_attrib["Height"])
        meta.width = int(volume_size_attrib["Width"])
        meta.depth = int(volume_size_attrib["Number_of_Frames"])

        scanning_params = root.find(".//Scanning_Parameters")
        scanning_params_attrib = scanning_params.attrib # type: ignore
        meta.bmscan = int(scanning_params_attrib["Number_of_BM_scans"])
        motor_pos = scanning_params_attrib.get("Motor_Pos")
        if motor_pos is not None:
            meta.motor_position = int(motor_pos)

        return meta

    # case no metadata request path to metadata or cancel file load
    else:
        return None

def unp_file_reader(path):
    """Take a path or list of paths to .unp files and return a list of LayerData tuples.

    Args:
        path(str or list of str): Path to file, or list of paths.

    Returns:
        layer_data : list of tuples
            A list of LayerData tuples where each tuple in the list contains
            (data, metadata, layer_type), where data is a numpy array, metadata is
            a dict of keyword arguments for the corresponding viewer.add_* method
            in napari, and layer_type is a lower-case string naming the type of
            layer. Both "meta", and "layer_type" are optional. napari will
            default to layer_type=="image" if not provided
    """

    # print(meta)
    meta = unp_proc_meta(path)
    if meta is None:
        show_info("No associated .ini or .xml meta data file found or process was cancelled. Cannot proceed.")
        return [(None,)]

    if meta.pattern == "Sine_Pause":
        display, display_hires = process_unp_sine_pause(Path(path), meta)
        _, tail = ospath.split(path)
        file_name = tail.split(".")[0]
        add_kwargs = {"name": file_name}
        layer_type = "image"
        bscan_layer = Layer.create(display, add_kwargs, layer_type)
        vmin, vmax = np.percentile(display, (1, 99))
        bscan_layer.contrast_limits = (float(vmin), float(vmax))

        add_kwargs_hires = {"name": file_name + "_hires"}
        layer_type_hires = "image"
        hires_layer = Layer.create(display_hires, add_kwargs_hires, layer_type_hires)
        vmin, vmax = np.percentile(display_hires, (1, 99))
        hires_layer.contrast_limits = (float(vmin), float(vmax))

        add_kwargs_enface = {"name": file_name + "_enface"}
        layer_type_enface = "image"
        if meta.max_projection:
            display_enface = np.max(display, axis=1)    
        else:
            display_enface = np.mean(display, axis=1)

        enface_layer = Layer.create(display_enface, add_kwargs_enface, layer_type_enface)
        vmin, vmax = np.percentile(display_enface, (1, 99))
        enface_layer.contrast_limits = (float(vmin), float(vmax))

        return [bscan_layer, hires_layer, enface_layer]
    
        #does no support sine pause with bmscan > 1 yet

    else:

        if meta.bmscan > 1:

            display = process_unp(Path(path), meta)

            if meta.octa != "none":
                _, tail = ospath.split(path)
                file_name = tail.split(".")[0]

                output_layers = []
                if meta.motor_position is not None:
                    add_kwargs = {"name": file_name+ "_raw", "metadata": {"motor_position" :meta.motor_position}}
                else:
                    add_kwargs = {"name": file_name+ "_raw"}
                layer_type = "image"
                bscan_layer = Layer.create(display, add_kwargs, layer_type)
                vmin, vmax = np.percentile(display, (1, 99))
                bscan_layer.contrast_limits = (float(vmin), float(vmax))
                output_layers.append(bscan_layer)

                if meta.structure:
                    #generate the strucutral and structural enface
                    if meta.motor_position is not None:
                        add_kwargs_structural = {"name": file_name + "_structural", "metadata": {"motor_position" :meta.motor_position}}
                    else:
                        add_kwargs_structural = {"name": file_name + "_structural"}
                    display_structural = average_bscans_torch(display, scans_per_avg=meta.bmscan)
                    structural_layer = Layer.create(display_structural, add_kwargs_structural, layer_type)
                    vmin, vmax = np.percentile(display_structural, (1, 99))
                    structural_layer.contrast_limits = (float(vmin), float(vmax))
                    output_layers.append(structural_layer)

                    add_kwargs_structural_enface = {"name": file_name + "_structural_enface"}
                    if meta.max_projection:
                        display_structural_enface = np.max(display_structural, axis=1)
                    else:
                        display_structural_enface = np.mean(display_structural, axis=1)

                    structural_enface_layer = Layer.create(display_structural_enface, add_kwargs_structural_enface, layer_type)
                    vmin, vmax = np.percentile(display_structural_enface, (1, 99))
                    structural_enface_layer.contrast_limits = (float(vmin), float(vmax))
                    output_layers.append(structural_enface_layer)


                #generate OCTA and OCTA enface
                if meta.motor_position is not None:
                    add_kwargs_octa = {"name": file_name + f"_OCTA_{meta.octa}", "metadata": {"motor_position" :meta.motor_position}}
                else:
                    add_kwargs_octa = {"name": file_name + f"_OCTA_{meta.octa}"}
                display_octa = generate_octa(
                    display,
                    mscans=meta.bmscan,
                    calc=OCTACalc[meta.octa],
                    )
                octa_layer = Layer.create(display_octa, add_kwargs_octa, layer_type)
                vmin, vmax = np.percentile(display_octa, (1, 99))
                octa_layer.contrast_limits = (float(vmin), float(vmax))
                output_layers.append(octa_layer)

                add_kwargs_octa_enface = {"name": file_name + f"_OCTA_{meta.octa}_enface"}
                if meta.max_projection:
                    display_octa_enface = np.max(display_octa, axis=1)
                else:
                    display_octa_enface = np.mean(display_octa, axis=1)

                octa_enface_layer = Layer.create(display_octa_enface, add_kwargs_octa_enface, layer_type)
                vmin, vmax = np.percentile(display_octa_enface, (1, 99))
                octa_enface_layer.contrast_limits = (float(vmin), float(vmax))  
                output_layers.append(octa_enface_layer)

                return output_layers
            
            else:
                _, tail = ospath.split(path)
                file_name = tail.split(".")[0]
                if meta.motor_position is not None:
                    add_kwargs = {"name": file_name, "metadata": {"motor_position" :meta.motor_position}}
                else:
                    add_kwargs = {"name": file_name}
                layer_type = "image"
                bscan_layer = Layer.create(display, add_kwargs, layer_type)
                vmin, vmax = np.percentile(display, (1, 99))
                bscan_layer.contrast_limits = (float(vmin), float(vmax))


                add_kwargs_enface = {"name": file_name + "_enface"}
                layer_type_enface = "image"

                if meta.max_projection:
                    display_enface = np.max(display, axis=1)
                else:
                    display_enface = np.mean(display, axis=1)

                enface_layer = Layer.create(display_enface, add_kwargs_enface, layer_type_enface)
                vmin, vmax = np.percentile(display_enface, (1, 99))
                enface_layer.contrast_limits = (float(vmin), float(vmax))

                return [bscan_layer, enface_layer]
        
        else:
            display = process_unp(Path(path), meta)

            _, tail = ospath.split(path)
            file_name = tail.split(".")[0]

            if meta.motor_position is not None:
                add_kwargs = {"name": file_name, "metadata": {"motor_position" :meta.motor_position}}
            else:
                add_kwargs = {"name": file_name}
            layer_type = "image"
            bscan_layer = Layer.create(display, add_kwargs, layer_type)
            vmin, vmax = np.percentile(display, (1, 99))
            bscan_layer.contrast_limits = (float(vmin), float(vmax))

            # else:
            add_kwargs_enface = {"name": file_name + "_enface"}
            layer_type_enface = "image"

            if meta.max_projection:
                display_enface = np.max(display, axis=1)
            else:
                display_enface = np.mean(display, axis=1)

            enface_layer = Layer.create(display_enface, add_kwargs_enface, layer_type_enface)
            vmin, vmax = np.percentile(display_enface, (1, 99))
            enface_layer.contrast_limits = (float(vmin), float(vmax))

            return [bscan_layer, enface_layer]