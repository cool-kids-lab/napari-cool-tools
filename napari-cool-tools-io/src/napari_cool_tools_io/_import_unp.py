from napari.utils.notifications import show_info
import ctypes
import sys

import os
import PyQt5.QtCore

from pathlib import Path

import multiprocessing

def convert_unp_prof():
    dirname = os.path.dirname(PyQt5.QtCore.__file__)
    plugin_path = os.path.join(dirname, 'plugins', 'platforms')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
    print(plugin_path)

    dll_filename = os.path.dirname(__file__) + "/UNPBatchProcessing.dll"
    print(dll_filename)

    my_lib = ctypes.WinDLL(dll_filename)

    #open the window (this will create an instance object)
    my_lib.showMainWindow()




    