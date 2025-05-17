import os
import os.path as ospath
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List
from enum import Enum
from pickle import HIGHEST_PROTOCOL

from napari.qt import thread_worker
from napari.layers import Image, Labels, Layer
from napari.types import FullLayerData
from napari.utils.notifications import show_info
from numpy import flip
from magicgui import magic_factory, widgets

from napari_cool_tools_io import torch, viewer


@thread_worker(progress=True)  # give us an indeterminate progress bar
def ndarray_to_file_thread(path, data, attributes, layer_type):
    """Thread wrapper around numpy.save() function
    Args:
        path(str or list of str): Path to file, or list of paths
        data(ndarray): Data from napari layer to be saved
    Returns:
        None saves ndarray data to .npy file designated by path
    """
    show_info("Pytorch .pt save thread has started.")
    print(f"data min/max: {data.min()}/{data.max()}\ndtype: {data.dtype}\n")
    out_data = torch.from_numpy(data.copy())
    out_layer_data_tuple = (out_data,attributes,layer_type)
    torch.save(out_layer_data_tuple,path,pickle_protocol=HIGHEST_PROTOCOL)
    #data.tofile(path)
    #np.save(path,data)
    show_info(".prof save thread has completed.")
    return


def torch_file_writer(path: str, layer_data: list[FullLayerData]) -> List[str]:
    """
    Args:
        path(str or list of str): Path to file, or list of paths
    Returns:

    """
    if len(layer_data) == 1:
        data, attributes, layer_type = layer_data[0]
        name = attributes["name"]

        show_info(f"path: {path}, shape: {data.shape}, attributes: {name}\n")

        # case .prof files should be 3 dimensional
        if data.ndim == 3:

            # reverse flip and transpose that occured upon loading
            save_data = flip(data, 1).transpose(0, 2, 1)
            worker = ndarray_to_file_thread(path, save_data, attributes, layer_type)
            worker.start()
            # save_data.tofile(path)

            show_info(f"Saving {path}")
    else:
        path = Path(path)
        p_dir = Path(path.parent) / path.stem
        ext = path.suffix
        os.makedirs(p_dir, exist_ok=True)

        for layer in layer_data:
            data, attributes, layer_type = layer
            name = attributes["name"]
            out_path = p_dir / f"{name}{ext}"

            show_info(
                f"path: {out_path}, shape: {data.shape}, attributes: {name}\n"
            )

            # case .prof files should be 3 dimensional
            if data.ndim == 3:
                dims = data.shape
                dtype = data.dtype

                # reverse flip and transpose that occured upon loading
                save_data = flip(data, 1).transpose(0, 2, 1)
                worker = ndarray_to_file_thread(out_path, save_data, attributes, layer_type)
                worker.start()
                # save_data.tofile(path)

                show_info(f"Saving {out_path}")

                # return [path]
            # case data is not proper dimension
            else:
                return None
    return [path]

@thread_worker(progress=True)
def torch_save_thread(data:any, path:Path):
    """"""
    show_info(f"torch save thread started!!\n")
    torch_data = torch.save(data,path,pickle_protocol=HIGHEST_PROTOCOL)
    show_info(f"torch save thread completed!!\n")
    return torch_data


class OutputType(Enum):
    """Enum for various output types."""
    Data = 1
    Dict = 2
    Img_Lbl_Pr = 3

def _on_init(widget):
    """"""
    widget.out_type.changed.connect(lambda val: torch_file_exporter_change(widget,val))

def torch_file_exporter_change(widget,val):
    """"""
    optional = ['key','img','lbls']
    #show_info(f"Output type in widget {widget} changed to: {val}\n\n")
    #show_info(f"{dir(widget)}\n")
    #show_info(f"{widget.asdict()}\n")
    #show_info(f"{widget.asdict().keys()}\n")
    #show_info(f"{widget.asdict().values()}\n")

    for k in optional:
        if k in widget.asdict().keys():
            #widget.remove(key)
            #show_info(f"key: {k}, widget: {widget[k]}\n")
            widget[k].visible = False
        else:
            pass

    # figure out how to hide widgets instead of dynamically creating and destroying them
    if val == OutputType.Data:
        pass
    elif val == OutputType.Dict:
        widget.key.visible = True
        #key = widgets.LineEdit(name = "key", label = "key", value='', annotation=str)
        #widget.append(key)
    elif val == OutputType.Img_Lbl_Pr:
        widget.file_name.visible = True
        widget.img_key.visible = True
        widget.lbl_key.visible = True
        widget.img.visible = True
        widget.lbls.visible = True
        #img = widgets.create_widget(name = "img", label = "img", value=None, annotation=Layer)
        #lbls = widgets.create_widget(name = "lbls", label = "lbls", value=None, annotation=Layer)
        #widget.append(img)
        #widget.append(lbls)


@magic_factory(
    widget_init=_on_init,
    out_type={"label": "output_type"},
    path={"label": ".pt output dir", "mode": "d"},
    key={"label": "key", "visible":False, "tooltip": "Saves data in dictionary with key"},
    file_name={"label": "file_name", "visible":False, "tooltip": "File name to save image-label data"},
    img_key={"label": "img_key", "visible":False, "tooltip": "Saves image data in dictionary with key"},
    lbl_key={"label": "lbl_key", "visible":False, "tooltip": "Saves label data in dictionary with key"},
    img={"label": "img", "visible":False},
    lbls={"label": "lbls", "visible":False},
    call_button="Export .pt file",
)
def torch_file_exporter(out_type: OutputType = OutputType.Data, path: Path = Path("./"), key: str="", file_name: str="image-labels", img_key: str="images", lbl_key: str="labels", img: Layer = None, lbls: Layer = None): 
#def torch_file_exporter(out_type: OutputType = OutputType.Data, path: Path = Path("./")): # use with remove widgets method
    """"""
    
    sel = list(viewer.layers.selection)

    if out_type == OutputType.Data or out_type == OutputType.Dict:
        for l in sel:
            data = l.data
            name = l.name
            out_path = path / f"{name}.pt"
            out_data = torch.from_numpy(data.copy())
            if out_type == OutputType.Data:
                #torch.save(out_data,out_path)
                worker = torch_save_thread(out_data,out_path)
                worker.start()
            elif out_type == OutputType.Dict:
                out_dict = {key: out_data}
                #torch.save(out_dict,out_path)
                worker = torch_save_thread(out_dict,out_path)
                worker.start()
    elif out_type == OutputType.Img_Lbl_Pr:
        #show_info(f"Not Yet Implemented!!\n")
        show_info(f"img layer: {img.name}, lbls layer: {lbls.name}\n")
        show_info(f"img layer: {img.data.shape}, lbls layer: {lbls.data.shape}\n")
        out_path = path / f"{file_name}.pt"
        image_data = img.data
        label_data = lbls.data
        image_data = torch.from_numpy(image_data.copy())
        label_data = torch.from_numpy(label_data.copy())
        out_dict = {
            img_key: image_data,
            lbl_key: label_data,
        }
        worker = torch_save_thread(out_dict,out_path)
        worker.start()

    return


def torch_file_importer(torch_file:Path)->List[Layer]:
    """"""
    torch_data = torch.load(torch_file)
    
    #show_info(f"{type(torch_data) == torch.Tensor}")
    file_name = torch_file.stem

    # if torch_data is a tensor load as image
    if type(torch_data) == torch.Tensor:
        show_info(f"Pytorch data contains a tensor.\n\nLoading data...\n\n{torch_data}\n")
        torch_data = torch_data.numpy()
        add_kwargs = {}
        layer_type = "image"
        layer = Layer.create(torch_data,add_kwargs,layer_type)
        return  [layer]
    elif type(torch_data) == dict:
        show_info(f"Pytorch data contains a dictionary.\n\nLoading data...\n\n{torch_data}\n")
        out_images = []
        for key, val in torch_data.items():
            if type(val) == torch.Tensor:
                val = val.numpy()
                show_info(f"Dictionary contains key('{key}') with value of type {type(val)}.\n")
                add_kwargs = {"name":f"{file_name}-{key}"}
                layer_type = "image"
                layer = Layer.create(val,add_kwargs,layer_type)
                #viewer.add_image(layer)
                out_images.append(layer)
        return out_images
    
    # else display contents via show_info
    else:
        show_info(f"Pytorch data is not a tensor or a dictionary.\n\nIt contains a {type(torch_data)}:\n\n{torch_data}\n")
        return [None]

