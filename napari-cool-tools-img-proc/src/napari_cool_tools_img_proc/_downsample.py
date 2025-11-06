from napari.layers import Image, Layer, Labels
import torch
from napari.qt.threading import thread_worker
from napari_cool_tools_io import viewer, device
from napari.utils.notifications import show_info

def downsample_image_plugin(
    img: Image,
    scale: float = 1
):
    """Downsample OCT volume for all dimensions.

    Args:
        vol (Image): 3D ndarray representing structural OCT data
        scale (float): scale factor for all dimensions

    Returns:
        Resized image

    """
    downsample_image_thread(img=img, scale=scale)

    return

@thread_worker(connect={"yielded": viewer.add_layer})
def downsample_image_thread(
    img: Image,
    scale: float = 1,
):
    show_info("Downsampling image")

    if len(img.data.shape) == 3:
        name = f"{img.name}_resized"
        layer_type = "image"

        ##############################
        #main function to resize using torch
        input_data = torch.Tensor(img.data).unsqueeze(0).unsqueeze(0).to(device)
        output_image = torch.nn.functional.interpolate(
            input_data, 
            scale_factor=(scale, scale, scale), 
            #size=target_size,        # explicit output size
            mode='trilinear', 
            align_corners=False
        )
        output_image = output_image.squeeze(0).squeeze(0).cpu().numpy()
        ######################################
        
        del input_data
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        add_kwargs = {"name": f"{name}"}
        layer = Layer.create(output_image, add_kwargs, layer_type)
        yield layer

        show_info("Resizing image thread has completed")

    else:
        show_info("Resizing image thread Failed, Must be a volume!!!")
