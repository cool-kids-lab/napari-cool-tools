import os
import os.path as ospath
import numpy as np
from napari.utils.notifications import show_info, show_error, show_warning
from scipy.io import loadmat
import h5py
from pathlib import Path
from magicgui import magic_factory
from napari.types import ImageData
from napari.layers import Image, Layer, Labels
import napari
import hdf5storage

def _on_init(widget):
    @widget.load_button.clicked.connect
    def do_load_button():

        octa = widget.octa.get_value()
        oct_segmentation = widget.oct_segmentation.get_value()

        octa_extension = octa.suffix
        oct_segmentation_extension = oct_segmentation.suffix

        if '.mat' not in octa_extension:
            show_error("Please select a mat file for the OCTA ")
            return

        if '.mat' not in oct_segmentation_extension:
            show_error("Please select a mat file for the Segmentation ")
            return

        octa_file = h5py.File(octa, 'r')

        ImgData = octa_file.get('InitPar').get('ImgData')
        if ImgData is None:
            show_warning("Could not find ImgData in the Mat File")
        ImgData = np.array(ImgData)
        seg_canvas = np.zeros_like(ImgData)
        ImgData = ImgData.transpose(0, 2, 1)
        ImgData = ImgData.astype(np.float64)

        add_kwargs = {"name":'ImgData'}
        layer_type = "image"
        struct_layer = Layer.create(ImgData,add_kwargs,layer_type)

        ssada = octa_file.get('InitPar').get('SSADA')
        if ssada is None:
            show_warning("Could not find SSADA in the Mat File")
        ssada = np.array(ssada)
        ssada = ssada.transpose(0, 2, 1)
        ssada = ssada.astype(np.float64)

        add_kwargs = {"name":'SSADA'}
        layer_type = "image"
        octa_layer = Layer.create(ssada,add_kwargs,layer_type)


        seg_file = h5py.File(oct_segmentation, 'r')
        seg_data = seg_file.get('ManualCurveData')
        if seg_data is None:
            show_warning("Could not find ManualCurveData in the Mat File")

        seg_data = np.array(seg_data)

        start_line = seg_data[:,0,:]
        stop_line = seg_data[:,5,:]

        for i in range(0,start_line.shape[0]):
            for j in range(0,start_line.shape[1]):
                start_idx = int(start_line[i,j])
                stop_idx = int(stop_line[i,j])
                seg_canvas[i,j,start_idx:stop_idx] = 1

        seg_canvas = seg_canvas.transpose(0, 2, 1)

        add_kwargs = {"name":'Segmentation'}
        layer_type = "labels"
        seg_layer = Layer.create(seg_canvas,add_kwargs,layer_type)

        viewer = napari.current_viewer()
        viewer.add_layer(struct_layer)
        viewer.add_layer(octa_layer)
        viewer.add_layer(seg_layer)

    @widget.save_button.clicked.connect
    def do_save_button():

        oct_segmentation = widget.oct_segmentation.get_value()
        oct_segmentation_extension = oct_segmentation.suffix

        if '.mat' not in oct_segmentation_extension:
            show_error("Please select a mat file for the Segmentation ")
            return
        
        if len(widget.segmentation_label) == 0:
            show_error("Please select a Label Layer")
            return

        viewer = napari.current_viewer()
        oct_label = viewer.layers[widget.segmentation_label.current_choice].data

        start_label = np.argmax(oct_label,axis=1) #(1200,1200)
        start_label = start_label.astype(np.uint16)
        start_label = start_label.transpose(1,0)

        oct_label = np.flip(oct_label,axis=1)

        stop_label = np.argmax(oct_label,axis=1) #(1200,1200)

        stop_label = oct_label.shape[1] - stop_label
        stop_label = stop_label.astype(np.uint16)
        stop_label = stop_label.transpose(1,0)
      
        filename = oct_segmentation.__str__()
        temp_filename = os.path.dirname(oct_segmentation) + '\\temp_dict.mat';

        seg_file = hdf5storage.loadmat(filename) 

        old_label = seg_file['ManualCurveData']

        old_label[:,0,:] =  start_label
        old_label[:,5,:] =  stop_label

        seg_file['ManualCurveData'] = old_label

        hdf5storage.savemat(temp_filename,seg_file, format='7.3',compression=False)
        os.remove(oct_segmentation)
        os.rename(temp_filename,filename)

        show_info("File Updated")


@magic_factory(
    # call_button="Load",
    call_button = False,
    widget_init=_on_init,
    load_button=dict(widget_type="PushButton", text="Load"),
    save_button=dict(widget_type="PushButton", text="Save"),
    octa={"label":"OCTA Mat File", "widget_type": "FileEdit", 'mode': 'r', 'filter': '*.mat'},
    oct_segmentation={"label":"Segmentation Mat File", "widget_type": "FileEdit", 'mode': 'r', 'filter': '*.mat'},
    segmentation_label={"label": "Label"}
    )

def load_segmentation_data(octa: Path, oct_segmentation : Path, load_button, segmentation_label: Labels , save_button):

    return