import pathlib
from pathlib import Path

import napari
import torch
from magicgui import magicgui

# import napari_cool_tools_io
from napari.utils.notifications import show_info

# from napari_cool_tools_oct_preproc import _oct_preproc_utils #generate_enface_image


@magicgui(
    proffolder={"label": "Input folder", "mode": "d"},
    call_button="Process .prof files",
)
def calc_gpu_batches(
    proffolder: pathlib.Path = Path("D:\\Mani\\BScan Labels 24\\Done\\591"),
):
    """ """
    # do something with the folders
    print(f"input folder: {proffolder}")

    # get list of .prof files
    prof_files = proffolder.glob("**/*.prof")
    # print(len(prof_files))

    # prof_files_list = list(prof_files)

    # create headless napari viewer
    viewer = napari.Viewer(show=False)

    # layer_cnt = len(viewer.layers) + 1

    file = list(prof_files)[0]
    viewer.open(file)
    # layer_cnt = len(viewer.layers) + 1
    # print(file, "loaded")
    # print(f"layer name is: {viewer.layers[0].name}")
    #prof_data = viewer.layers[0]

    # get gpu propeerties
    gpu_prop = torch.cuda.get_device_properties(0)
    show_info(f"\nGPU properties: {gpu_prop}\n")
    # get gpu memory
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    show_info(f"\nGPU memory available: {gpu_mem}\n")

    layer = viewer.layers[0]
    layer_data_tuple = layer.as_layer_data_tuple()
    data, kwargs, layer_type = layer_data_tuple
    name = kwargs["name"]
    show_info(f"\n\n{layer_type} layer {name} contains data of shape {data.shape}\n\n")

    show_info(f"\n\nA single B-scan is size {data[0].nbytes} bytes\n\n")
    show_info(
        f"\n\nEach B-scan has {data[0].size} items that are {data[0].itemsize} bytes large for {data[0].size * data[0].itemsize} bytes per B-scan\n\n"
    )
    show_info(
        f"\n\n{gpu_mem / data[0].nbytes} B-scans can fit into gpu memory at once.\n\n"
    )

    """'
    for file in prof_files:
        #print(file)

        #while len(viewer.layers) == layer_cnt:
        #    layer_cnt = len(viewer.layers)
        viewer.open(file)
        #layer_cnt = len(viewer.layers) + 1
        print(file, "loaded")
        print(f"layer name is: {viewer.layers[0].name}")
        prof_data = viewer.layers[0]

        #viewer.window.add_plugin_function()
        #viewer.window.add_plugin_function(generate_enface_image(prof_data, debug=False, sin_correct=True, log_correct=True, band_pass_filter=False, CLAHE=False))

        #print(f"enface layer: {viewer.layers[1].name}")
        
    """

    # viewer.open(prof_files_list)
    # print(len(prof_files))


# print(dir(viewer_model))
# process_files.show(run=True)
calc_gpu_batches()
