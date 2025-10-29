from napari.layers import Image, Layer, Labels
import torch
from napari.qt.threading import thread_worker
from napari_cool_tools_io import viewer, device
from napari.utils.notifications import show_info

def resize_image_plugin(
    img: Image,
    xscale: float = 1,
    yscale: float = 1,
    zscale: float = 1,
):
    """resize OCT volume.

    Args:
        vol (Image): 3D ndarray representing structural OCT data
        xscale (float): scale in x direction
        yscale (float): scale in y direction
        zscale (float): scale in z direction

    Returns:
        Resized image

    """
    resize_image_thread(img=img, xscale=xscale,yscale=yscale, zscale=zscale)

    return

@thread_worker(connect={"yielded": viewer.add_layer})
def resize_image_thread(
    img: Image,
    xscale: float = 1,
    yscale: float = 1,
    zscale: float = 1,
):
    show_info("Resizing image")

    if len(img.data.shape) == 3:
        name = f"{img.name}_resized"
        layer_type = "image"

        ##############################
        #main function to resize using torch
        input_data = torch.Tensor(img.data).unsqueeze(0).unsqueeze(0).to(device)
        output_image = torch.nn.functional.interpolate(
            input_data, 
            scale_factor=(zscale, yscale, xscale), 
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
