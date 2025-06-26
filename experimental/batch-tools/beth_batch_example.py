""" """

from pathlib import Path

# from qtpy.QtWidgets import QApplication
import napari
from magicgui import magicgui

# import pptk
# import open3d as o3d
# from pypcd4 import PointCloud
# from napari_cool_tools_io
from napari_cool_tools_img_proc import DType
from napari_cool_tools_img_proc._equalization_funcs import init_bscan_preproc

# from napari_cool_tools_vol_proc._projection_tools import mip
from napari_cool_tools_oct_preproc._oct_preproc_utils_funcs import generate_enface
from napari_cool_tools_segmentation import EnfaceSegmentationType
from napari_cool_tools_segmentation._segmentation_funcs import enface_onnx_seg_func
from tqdm import tqdm


@magicgui(
    fold_dir={"label": "Fold Directory", "mode": "d"},
    output_dir={"label": "Output Directory", "mode": "d"},
    call_button="Generate Training Folds",
)
def generate_enface_with_labels(
    fold_dir: Path = Path(r"D:\JJ\Projects\Segmentation_Paper\Data\Bscan"),
    output_dir: Path = Path(r"D:\JJ\Projects\Segmentation_Paper\Data\Bscan"),
):
    """"""

    file_paths = list(fold_dir.rglob("*_Images.pt"))
    #test_file_path = file_paths[0]

    viewer = napari.Viewer(show=False)

    for _, file_path in tqdm(enumerate(file_paths), desc="Processing OCT Volumes"):
        # viewer.open(test_file_path,plugin="napari-cool-tools-io")
        viewer.open(file_path, plugin="napari-cool-tools-io")
        oct_data_layer = viewer.layers[-1]

        data = viewer.layers[-1].data
        init_data = init_bscan_preproc(
            data, num_std=16, min_intensity=0.0, max_intensity=1.0, dtype=DType.NP_FLOAT
        )

        viewer.layers.remove(oct_data_layer)

        viewer.add_image(init_data)
        init_oct_data_layer = viewer.layers[-1]
        # mip_data = mip

        onnx_enface_ridge = EnfaceSegmentationType.RIDGE.value

        enface_data = list(
            generate_enface(
                data,
                sin_correct=False,
                CLAHE=True,
                clahe_clip=2.5,
                log_correct=True,
                log_gain=1.0,
            )
        )[0][0]
        init_enface_data = list(
            generate_enface(
                init_data,
                sin_correct=False,
                CLAHE=True,
                clahe_clip=2.5,
                log_correct=True,
                log_gain=1.0,
            )
        )[0][0]
        ridge_labels = enface_onnx_seg_func(
            enface_data,
            onnx_path=onnx_enface_ridge,
            segmentation_type="ridge",
            label_val=4,
            use_cpu=True,
            blur=False,
        )
        init_ridge_labels = enface_onnx_seg_func(
            init_enface_data,
            onnx_path=onnx_enface_ridge,
            segmentation_type="ridge",
            label_val=4,
            use_cpu=True,
            blur=False,
        )

        viewer.layers.remove(init_oct_data_layer)
        # print(type(enface_data))
        # print(enface_data)
        # print(enface_data.shape)
        viewer.add_image(enface_data)
        viewer.add_labels(ridge_labels)
        viewer.add_image(init_enface_data)
        viewer.add_labels(init_ridge_labels)

    print("Batch Processing Complete.")

    viewer.show()
    napari.run()

    # o3d.visualization.draw_geometries([o3d_pcd])


# view_bscan_variants.changed.connect(print)
# app = QApplication([])
generate_enface_with_labels.show(run=True)
# app.exec_()
