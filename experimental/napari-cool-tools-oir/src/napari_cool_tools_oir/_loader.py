import os
import numpy as np
from napari.utils.notifications import show_info, show_error, show_warning
from scipy.io import loadmat
import h5py
from pathlib import Path
from magicgui import magic_factory
from napari.layers import Image, Layer, Labels
import napari
import torchvision.transforms.functional as F
import torch
from scipy.io import savemat
import tifffile

def _on_init(widget):
    @widget.load_button.clicked.connect
    def do_load_button():

        octa = widget.octa.get_value()

        octa_extension = octa.suffix

        if '.mat' not in octa_extension:
            show_error("Please select a mat file for the OCTA ")
            return

        octa_file = h5py.File(octa, 'r')

        ImgData = octa_file.get('InitPar').get('ImgData')
        if ImgData is None:
            show_warning("Could not find ImgData in the Mat File")
        ImgData = np.array(ImgData)
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

        viewer = napari.current_viewer()
        viewer.add_layer(struct_layer)
        viewer.add_layer(octa_layer)

    @widget.predict_button.clicked.connect
    def do_predict_button():

        dnn_filename = os.path.dirname(__file__) + "/traced_model.pt"
        model = torch.jit.load(dnn_filename)
        model.eval()
        model.cpu()

        if torch.cuda.is_available():
            model.cuda()


        with torch.no_grad():
            viewer = napari.current_viewer()
            input_image = viewer.layers[widget.input_image.current_choice].data
            output_image =np.zeros_like(input_image)   

            counter = 0
            for in_image in input_image:

                mmin = in_image.min()
                mmax = in_image.max()
                delta = mmax-mmin
                
                in_image = in_image - mmin
                in_image = in_image / delta

                in_image = torch.from_numpy(in_image)
                in_image = in_image.cpu()
                in_image = torch.unsqueeze(in_image, 0)
                resized_in_image  = F.resize(in_image,size = (672,1200))
                resized_in_image = torch.unsqueeze(resized_in_image, 0)
                
                resized_in_image = resized_in_image.float()

                if torch.cuda.is_available():
                    resized_in_image = resized_in_image.cuda()

                out_image = model(resized_in_image)
                resized_out_image  = F.resize(out_image,size = (in_image.shape[-2],in_image.shape[-1]))
                resized_out_image = resized_out_image[0].detach().squeeze().cpu().numpy()
                output_image[counter] = resized_out_image
                counter = counter +1

        output_image = np.round(output_image)
        output_image = output_image.astype(np.int32)
        add_kwargs = {"name":widget.input_image.current_choice+"_output"}
        layer_type = "labels"
        out_layer = Layer.create(output_image,add_kwargs,layer_type)
        
        viewer.add_layer(out_layer)

        torch.cuda.empty_cache()



    @widget.save_button.clicked.connect
    def do_save_button():
       
        if len(widget.segmentation_label) == 0:
            show_error("Please select a Label Layer")
            return

        viewer = napari.current_viewer()
        oct_label = viewer.layers[widget.segmentation_label.current_choice].data

        octa_filename = widget.octa.get_value()

        octa_filename = str(octa_filename)

        output_name = octa_filename[:-4] + "_seg.mat"

        savemat(output_name, {'segmentation': oct_label})

        output_name = octa_filename[:-4] + "_seg.tif"

        tifffile.imwrite(output_name,oct_label)

        show_info("File saved in the same directory")


@magic_factory(
    # call_button="Load",
    call_button = False,
    widget_init=_on_init,
    load_button=dict(widget_type="PushButton", text="Load"),
    predict_button=dict(widget_type="PushButton", text="Predict"),
    save_button=dict(widget_type="PushButton", text="Save"),
    octa={"label":"OCTA Mat File", "widget_type": "FileEdit", 'mode': 'r', 'filter': '*.mat'}
    )
def load_segmentation_data(octa: Path, load_button, input_image: Image, predict_button, segmentation_label: Labels, save_button):

    return
    
