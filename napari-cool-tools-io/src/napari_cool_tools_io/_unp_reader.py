import os
import os.path as ospath
import xml.etree.ElementTree as ET
from pathlib import Path
import configparser
import numpy as np
from napari.utils.notifications import show_info
from napari_cool_tools_io.process_unp import process_unp, process_unp_sine_pause
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout
from qtpy.QtWidgets import QCheckBox
from qtpy.QtWidgets import QDialog, QDialogButtonBox
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from napari_cool_tools_io import unp_meta

def xml_dialog(parent=None, double_side_button=True, full_range_button=True, auto_dispersion_button=True, desine_button=True):
    """Show a modal dialog with a 'Double-sided' checkbox and Accept/Cancel buttons.

    Returns:
        tuple[str, bool]: ("accepted" or "canceled", final_double_side_value)
    """
    # local imports to avoid changing top-level imports

    dialog = QDialog(parent)
    dialog.setWindowTitle("Double Side Option")

    double_side_checkbox = QCheckBox("Double Sided")
    double_side_checkbox.setChecked(False)

    full_range_checkbox = QCheckBox("Full Range")
    full_range_checkbox.setChecked(False)

    auto_dispersion_checkbox = QCheckBox("Auto Dispersion Compensation")
    auto_dispersion_checkbox.setChecked(False)

    desine_checkbox = QCheckBox("Desine")
    desine_checkbox.setChecked(False)

    btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    # connect the button box signals so the dialog will accept/reject and close
    btn_box.accepted.connect(dialog.accept)
    btn_box.rejected.connect(dialog.reject)

    main_layout = QVBoxLayout()
    if double_side_button:
        main_layout.addWidget(double_side_checkbox)

    if full_range_button:
        main_layout.addWidget(full_range_checkbox)

    if auto_dispersion_button:
        main_layout.addWidget(auto_dispersion_checkbox)

    if desine_button:
        main_layout.addWidget(desine_checkbox)

    main_layout.addWidget(btn_box)
    dialog.setLayout(main_layout)

    result = dialog.exec_()

    if result == QDialog.Accepted:
        return (True, bool(double_side_checkbox.isChecked()), bool(full_range_checkbox.isChecked()),
                 bool(auto_dispersion_checkbox.isChecked()), bool(desine_checkbox.isChecked()))
    else:
        # checkbox has been reverted in _on_reject, so return that value
        return (False, bool(double_side_checkbox.isChecked()), bool(full_range_checkbox.isChecked()),
                 bool(auto_dispersion_checkbox.isChecked()), bool(desine_checkbox.isChecked()))

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
        meta.vista = config.getint('Scanning', 'VISTA_Num')
        meta.packed = config.getboolean('Acquisition', 'PACKED12')
        meta.double_side = config.getboolean('Scanning', 'Bidirectional')
        meta.pattern = config['Scanning']['Pattern']
        meta.delay = config.getint('Scanning', 'XDelay')

        if meta.pattern == "Sine_Pause":
            meta.sine_frame_indices = list(map(int, config['Scanning']['Sine_Pause_Frame_Index'].split()))
            meta.sine_hires_ratio = config.getint('Scanning', 'Sine_Pause_X_Rate_Reduction')

        status, _, meta.full_range, meta.auto_dispersion, meta.desine = xml_dialog(double_side_button=False)

        if not status:
            return None

        print("File Info")
        print(f"width: {meta.width}")
        print(f"height: {meta.height}")
        print(f"depth: {meta.depth}")
        print(f"bmscan: {meta.bmscan}")
        print(f"vista: {meta.vista}")
        print(f"packed: {meta.packed}")
        print(f"double_side: {meta.double_side}")
        print(f"pattern: {meta.pattern}")
        print(f"delay: {meta.delay}")

        return meta

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

        status, meta.double_side, meta.full_range, meta.auto_dispersion, meta.desine = xml_dialog()
        
        print("File Info")
        print(f"width: {meta.width}")
        print(f"height: {meta.height}")
        print(f"depth: {meta.depth}")
        print(f"bmscan: {meta.bmscan}")
        print(f"vista: {meta.vista}")
        print(f"packed: {meta.packed}")
        print(f"double_side: {meta.double_side}")
        print(f"pattern: {meta.pattern}")
        print(f"delay: {meta.delay}")

        if not status:
            return None
        
        #packed is always false for xml only case
        meta.packed = False

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

    meta = unp_proc_meta(path)
    if meta is None:
        show_info("No associated .ini or .xml meta data file found or process was cancelled. Cannot proceed.")
        return None
    

    #TODO: Handle Vista Scans properly

    #TODO: Handle OCTA
    # #does not support Sine_Pause pattern at the moment
    if meta.pattern == "Sine_Pause":
        display, display_hires = process_unp_sine_pause(Path(path), meta)

        display = display.transpose(0,2,1)  # change from (depth, height, width) to (depth, width, height) for napari
        display_hires = display_hires.transpose(0,2,1)  # change from (depth, height, width) to (depth, width, height) for napari
        
        head, tail = ospath.split(path)
        file_name = tail.split(".")[0]
        add_kwargs = {"name": file_name}
        layer_type = "image"

        add_kwargs_hires = {"name": file_name + "_hires"}
        layer_type_hires = "image"

        return [(display, add_kwargs, layer_type), (display_hires, add_kwargs_hires, layer_type_hires)]
    
    else:

        display = process_unp(Path(path), meta)

        display = display.transpose(0,2,1)  # change from (depth, height, width) to (depth, width, height) for napari

        head, tail = ospath.split(path)
        file_name = tail.split(".")[0]
        add_kwargs = {"name": file_name}
        layer_type = "image"

        return [(display, add_kwargs, layer_type)]
