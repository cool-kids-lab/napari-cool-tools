try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = ()

import napari
import torch
from napari.utils.notifications import show_info

viewer = napari.current_viewer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def memory_stats():
    show_info(f"Gpu memory allocated: {torch.cuda.memory_allocated()/1024**2}")
    show_info(f"Gpu memory reserved: {torch.cuda.memory_reserved()/1024**2}")
    
    gpu_mem_clear = (torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0)

    print(f"GPU memory is clear: {gpu_mem_clear}\n")
    if not gpu_mem_clear:
        print(f"{torch.cuda.memory_summary()}\n")


@napari.Viewer.bind_key("i")
def shortcut(viewer):
    # init layers and active selection
    layers = viewer.layers
    curr_sel = layers.selection

    # check that label layer is selected

    # case only one layer is selected
    if len(curr_sel) == 1:
        # get current layer
        curr_layer = list(curr_sel)[0]
        curr_layer_type = curr_layer.as_layer_data_tuple()[2]

        # set default opacity
        new_opacity = curr_layer.opacity

        # case selected layer is a labels layer
        if curr_layer_type == "labels":
            # get current opacity
            opacity = curr_layer.opacity

            # case opacity is greater than 0
            if opacity > 0:
                # increase opacity size
                new_opacity = opacity - 0.1

                if new_opacity < 0:
                    new_opacity = 0
                else:
                    pass

                curr_layer.opacity = new_opacity

                # update viewer with mesage
                msg = f"decrease opacity to {new_opacity}"
                viewer.status = msg

            # case opacity is < 0
            else:
                pass

        # case selected layer is not a labels layer
        else:
            pass

        pass

    # case multiple layers are selected
    else:
        pass


@napari.Viewer.bind_key("o")
def shortcut2(viewer):
    # init layers and active selection
    layers = viewer.layers
    curr_sel = layers.selection

    # check that label layer is selected

    # case only one layer is selected
    if len(curr_sel) == 1:
        # get current layer
        curr_layer = list(curr_sel)[0]
        curr_layer_type = curr_layer.as_layer_data_tuple()[2]

        # set default opacity
        new_opacity = curr_layer.opacity

        # case selected layer is a labels layer
        if curr_layer_type == "labels":
            # get current opacity
            opacity = curr_layer.opacity

            # case opacity is less than 1
            if opacity < 1:
                # increase opacity
                new_opacity = opacity + 0.1

                if new_opacity > 1:
                    new_opacity = 1
                else:
                    pass

                curr_layer.opacity = new_opacity

                # update viewer with mesage
                msg = f"increase opacity to {new_opacity}"
                viewer.status = msg

            # case opacity is >= 1
            else:
                pass

        # case selected layer is not a labels layer
        else:
            pass

        pass

    # case multiple layers are selected
    else:
        pass


@napari.Viewer.bind_key("[")
def shortcut3(viewer):
    # init layers and active selection
    layers = viewer.layers
    curr_sel = layers.selection

    # check that label layer is selected

    # case only one layer is selected
    if len(curr_sel) == 1:
        # get current layer
        curr_layer = list(curr_sel)[0]
        curr_layer_type = curr_layer.as_layer_data_tuple()[2]

        # case selected layer is a labels layer
        if curr_layer_type == "labels":
            # get current brush size
            brush_size = curr_layer.brush_size

            # case brush size is greater than 1
            if brush_size > 1:
                # case brush size is odd
                if brush_size % 2 == 1:
                    brush_size = brush_size - 1

                # case brush size is even
                else:
                    pass

                # decrease brush size
                curr_layer.brush_size = brush_size - 1

                # update viewer with mesage
                msg = f"decrease brush size to {brush_size-1}"
                viewer.status = msg

            # case brush size is <= 1
            else:
                pass

        # case selected layer is not a labels layer
        else:
            pass

        pass

    # case multiple layers are selected
    else:
        pass


@napari.Viewer.bind_key("]")
def shortcut4(viewer):
    # init layers and active selection
    layers = viewer.layers
    curr_sel = layers.selection

    # check that label layer is selected

    # case only one layer is selected
    if len(curr_sel) == 1:
        # get current layer
        curr_layer = list(curr_sel)[0]
        curr_layer_type = curr_layer.as_layer_data_tuple()[2]

        # case selected layer is a labels layer
        if curr_layer_type == "labels":
            # get current brush size
            brush_size = curr_layer.brush_size

            # case brush size is less than 40
            if brush_size < 40:
                # case brush size is even
                if brush_size % 2 == 0:
                    brush_size = brush_size + 1

                # case brush size is odd
                else:
                    pass

                # increase brush size
                curr_layer.brush_size = brush_size + 1

                # update viewer with mesage
                msg = f"increase brush size to {brush_size+1}"
                viewer.status = msg

            # case brush size is >= 40
            else:
                pass

        # case selected layer is not a labels layer
        else:
            pass

        pass

    # case multiple layers are selected
    else:
        pass
