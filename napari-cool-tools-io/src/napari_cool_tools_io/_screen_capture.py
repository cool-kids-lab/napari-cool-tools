from magicgui import magic_factory
from napari.layers import Layer
from napari.utils.notifications import show_info

from napari_cool_tools_io import viewer


@magic_factory(call_button="Data Capture from View")
# @magic_factory(auto_call=True)
def data_capture_from_view(selection=viewer.layers.selection) -> Layer:
    """"""

    # case active selection
    if selection.active is not None:
        # print(f'\n\nselected layer:{selection}\n\n')
        layer = selection.active

        name = layer.name

        # optional layer type argument
        layer_type = "image"

        # get_current_slice.selected.value = layer
        # get_current_slice.show()
        # case 3D data

        if (layer.data.squeeze().ndim == 3 and layer.rgb is False) or (
            layer.data.squeeze().ndim == 4 and layer.rgb is True
        ):
            depth_idx = viewer.dims.order[0]
            current_depth = viewer.dims.current_step[depth_idx]
            viewed_data = layer.data.transpose(viewer.dims.order)[
                current_depth
            ]

            print(
                f"\nGetting Numpy slice...\nDepth: {current_depth}\nAlong data axis: {depth_idx}\nLayer: {layer}\nDimensions: {viewed_data.shape}\n"
            )

            # optional kwargs for viewer.add_* method
            add_kwargs = {"name": f"{name}_idx_{current_depth}_data"}

            output = Layer.create(viewed_data, add_kwargs, layer_type)

            return output

        else:
            if layer.rgb is False:
                show_info(
                    f"\nNon-RGB data in selected layer must have 3 dimensions not {layer.data.squeeze().ndim} dimensions\n"
                )
            else:
                show_info(
                    f"\nRGB data in selected layer must have 4 dimensions not {layer.data.squeeze().ndim} dimensions\n"
                )

    else:
        show_info(
            "\nA single layer needs to be selected for this function to work\n"
        )


@magic_factory(call_button="Image Capture from View")
def image_capture_from_view(selection=viewer.layers.selection) -> Layer:
    """"""

    # print(f'\n\ncurrent selection: {selection.active}\n\n')
    # print(dir(ndarray))
    # print(f'\n\nlist version:\n{list(ndarray)}\n\n')
    ## case 3D data
    # if ndarray.ndim == 3:
    #    print(f'\n\nselected layer:{ndarray}\n\n')
    ## case not 3D data
    # else:
    #    show_info(f'')

    # viewer.layers.selection.events.active.connect(current_selection)

    # print(dir(magic_factory))

    # case active selection
    if selection.active is not None:
        # print(f'\n\nselected layer:{selection}\n\n')
        layer = selection.active

        name = layer.name

        # optional layer type argument
        layer_type = "image"

        # get_current_slice.selected.value = layer
        # get_current_slice.show()
        # case 3D data

        if (layer.data.squeeze().ndim == 3 and layer.rgb is False) or (
            layer.data.squeeze().ndim == 4 and layer.rgb is True
        ):
            depth_idx = viewer.dims.order[0]
            current_depth = viewer.dims.current_step[depth_idx]
            viewed_data = layer.data.transpose(viewer.dims.order)[
                current_depth
            ]

            print(
                f"\nGetting slice...\nDepth: {current_depth}\nAlong data axis: {depth_idx}\nLayer: {layer}\nDimensions: {viewed_data.shape}\n"
            )

            # normalize data by clims
            normalized_data = (viewed_data - layer.contrast_limits[0]) / (
                layer.contrast_limits[1] - layer.contrast_limits[0]
            )
            colormapped_data = layer.colormap.map(normalized_data.flatten())
            pre_gamma = colormapped_data.reshape(normalized_data.shape + (4,))

            # viewer.add_image(viewed_slice)
            # viewer.add_image(gamma_corrected)

            from skimage import exposure

            viewed_slice = exposure.adjust_gamma(
                pre_gamma, layer.gamma
            )  # investigat napari vs skimage gamma as the results differ slightly
            # optional kwargs for viewer.add_* method
            add_kwargs = {"name": f"{name}_idx_{current_depth}_screenshot"}
            output = Layer.create(viewed_slice, add_kwargs, layer_type)
            # viewer.add_image(viewed_slice)
            return output

        else:
            if layer.rgb is False:
                show_info(
                    f"\nNon-RGB data in selected layer must have 3 dimensions not {layer.data.squeeze().ndim} dimensions\n"
                )
            else:
                show_info(
                    f"\nRGB data in selected layer must have 4 dimensions not {layer.data.squeeze().ndim} dimensions\n"
                )

    else:
        show_info(
            "\nA single layer needs to be selected for this function to work\n"
        )
