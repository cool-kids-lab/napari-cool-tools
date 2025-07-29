""" """

from pathlib import Path
from typing import Literal

# from qtpy.QtWidgets import QApplication
import numpy as np
import napari
from magicgui import magicgui

# import pptk
# import open3d as o3d
# from pypcd4 import PointCloud
from napari_cool_tools_io._prof_reader import prof_proc_meta
from napari_cool_tools_img_proc import DType
from napari_cool_tools_vol_proc import ProjectionType
from napari_cool_tools_oct_preproc._oct_preproc_utils_funcs import preproc_bscan
from napari_cool_tools_img_proc._equalization_funcs import init_bscan_preproc

# from napari_cool_tools_vol_proc._projection_tools import mip
from napari_cool_tools_oct_preproc._oct_preproc_utils_funcs import generate_enface
from napari_cool_tools_oct_preproc import Preproc

from napari_cool_tools_vol_proc._averaging_tools_funcs import average_bscans, average_per_bscan

from napari_cool_tools_segmentation import EnfaceSegmentationType
from napari_cool_tools_segmentation._segmentation_funcs import enface_onnx_seg_func, bscan_onnx_seg_func, bscan_onnx_deconj_func
from tqdm import tqdm


@magicgui(
    fold_dir={"label": "Fold Directory", "mode": "d"},
    output_dir={"label": "Output Directory", "mode": "d"},
    call_button="Generate Training Folds",
)
def generate_enface_with_labels(
    fold_dir: Path = Path(r"D:\JJ\Projects\Beth_Automation\test_batch"),
    output_dir: Path = Path(r"D:\JJ\Projects\Beth_Automation\test_batch_save_data"),
    preproc_suffix:str = "cor_preproc_bscan",
    ret_chor_suffix:str = "ret_chor_seg",
    max_proj_suffix:str = "max_inten_proj",
    mean_proj_suffix:str = "mean_inten_proj",
    en_face_labels_suffix:str = "en_face_labels",
    en_face_ridge_suffix:str = "en_face_ridge_labels",
    en_face_ON_suffix:str = "en_face_on_labels",
    en_face_vessel_suffix:str = "en_face_vessel_labels",
    img_dtype:DType = DType.NP_FLOAT32,
    label_dtype:DType = DType.NP_UINT8,
    oct_type: Literal["OCT","OCTA"] = "OCT",
    deconjugate:bool = True,
    desine:bool = False,
    log_cor:bool = True,
    init_bscan_proc:bool=True,
    bscan_batch_size:int=32,
    bscan_use_cpu:bool = False,
    enface_max_proj:bool = True,
    enface_mean_proj:bool = True,
    enface__proj:bool = True,
    save_enface_vessel:bool = False,
    save_enface_optic_nerve:bool = True,
    save_enface_ridge:bool = True,
    save_pre_proc:bool = True,
    show_in_napari:bool = False,
    #enface_projection:ProjectionType=ProjectionType.ARGMAX,
    debug:bool = False,
):
    """"""

    #file_paths = list(fold_dir.rglob("*_processed.prof"))
    file_paths = list(fold_dir.rglob("*.prof"))
    #test_file_path = file_paths[0]

    if show_in_napari:
        viewer = napari.Viewer(show=False)

    for _, file_path in tqdm(enumerate(file_paths), desc="Processing OCT Volumes"):
        # viewer.open(test_file_path,plugin="napari-cool-tools-io")
        print(type(file_path))
        name = Path(file_path).stem

        # load prof

        if debug:
            print("Loading Data")

        if show_in_napari:
            viewer.open(file_path, plugin="napari-cool-tools-io")
            oct_data_layer = viewer.layers[-1]

            data = viewer.layers[-1].data

        meta = prof_proc_meta(file_path, ".prof")
        h, w, d, bmscan, w_param, dtype, layer_type = meta

        dot_prof = np.dtype(("<f4", (h, w)))
        data = np.fromfile(file_path, dtype=dot_prof, count=-1)
        data = np.flip(data.transpose(0, 2, 1), 1)

        if debug:
            print("Data Loaded")


        
        if deconjugate:
            # deconjugate prof
            if debug:
                print("Deconjugating Data")
            data,_ = bscan_onnx_deconj_func(data)

            if debug:
                print("Deconjugating complete")


        # preprocess prof
        if debug:
            print("Preprocessing Data")

        if oct_type == "OCT":
            data = average_per_bscan(data,scans_per_avg=3,axis=0,trim=False)
        elif oct_type == "OCTA":
            data = average_bscans(data,scans_per_avg=3)

        preproc_data = preproc_bscan(data,Preproc.SN,bg_rm_ct_adj=init_bscan_proc,ascan_corr=desine,log_cor=log_cor,vol_proc=True,processor='cpu')

        if debug:
            print("Saving Preproc Data")

        # preproc_name = f"{name}_{preproc_suffix}.npy"
        # preproc_path = output_dir/preproc_name
        # np.save(preproc_path,preproc_data.astype(img_dtype.value))

        if debug:
            print("Preproc Data saved")

        # init_data = init_bscan_preproc(
        #     preproc_data, num_std=16, min_intensity=0.0, max_intensity=1.0, dtype=DType.NP_FLOAT64
        # )

        if debug:
            print("Preprocessing Data complete")

        #viewer.layers.remove(oct_data_layer)

        # segment prof bscans
        if debug:
            print("Segmenting Data")
        #ret_chor_labels,_ = bscan_onnx_seg_func(data,batch_size=bscan_batch_size,use_cpu=bscan_use_cpu)[0]
        ret_chor_labels,_ = bscan_onnx_seg_func(preproc_data,batch_size=bscan_batch_size,use_cpu=bscan_use_cpu)[0]

        if debug:
            print("Saving ret chor labels")

        ret_chor_name = f"{name}_{ret_chor_suffix}.npy"
        ret_chor_name_path = output_dir/ret_chor_name
        np.save(ret_chor_name_path,ret_chor_labels.astype(label_dtype.value))

        if debug:
            print("Ret chor labels saved")

        if debug:
            print("Segmenting Data complete")

        if show_in_napari:
            viewer.add_image(preproc_data)
            viewer.add_labels(ret_chor_labels)

        #init_oct_data_layer = viewer.layers[-1]
        # mip_data = mip

        # generate en face and segment

        # enface_data = list(
        #     generate_enface(
        #         data,
        #         sin_correct=False,
        #         CLAHE=True,
        #         clahe_clip=2.5,
        #         log_correct=True,
        #         log_gain=1.0,
        #     )
        # )[0][0]
        if debug:
            print("Generating en face Data")

        if enface_mean_proj:
            init_enface_data = list(
                generate_enface(
                    preproc_data,
                    projection_type=ProjectionType.MEAN,
                    sin_correct=False,
                    CLAHE=True,
                    clahe_clip=2.5,
                    log_correct=True,
                    log_gain=1.0,
                )
            )[0][0]

            if debug:
                print("Saving mean intensity proj")

            enface_mean_name = f"{name}_{mean_proj_suffix}.npy"
            enface_mean_name_path = output_dir/enface_mean_name
            np.save(enface_mean_name_path,init_enface_data.astype(img_dtype.value))

        if debug:
            print("Mean intensity saved")

        if enface_max_proj:
            init_enface_data = list(
                generate_enface(
                    preproc_data,
                    projection_type=ProjectionType.MAX,
                    sin_correct=False,
                    CLAHE=True,
                    clahe_clip=2.5,
                    log_correct=True,
                    log_gain=1.0,
                )
            )[0][0]

            if debug:
                print("Saving max intensity proj")

            enface_max_name = f"{name}_{max_proj_suffix}.npy"
            enface_max_name_path = output_dir/enface_max_name
            np.save(enface_max_name_path,init_enface_data.astype(img_dtype.value))

            if debug:
                print("Max intensity saved")

        # ridge_labels = enface_onnx_seg_func(
        #     enface_data,
        #     onnx_path=onnx_enface_ridge,
        #     segmentation_type="ridge",
        #     label_val=4,
        #     use_cpu=True,
        #     blur=False,
        # )
        if save_enface_ridge:

            if debug:
                print("Saving ridge labels")
            init_ridge_labels = enface_onnx_seg_func(
                init_enface_data,
                onnx_path=EnfaceSegmentationType.RIDGE.value,
                segmentation_type="ridge",
                label_val=4,
                use_cpu=True,
                blur=False,
            )
            ridge_labels_name = f"{name}_{en_face_ridge_suffix}.npy"
            ridge_labels_name_path = output_dir/ridge_labels_name
            np.save(ridge_labels_name_path,init_ridge_labels.astype(label_dtype.value))

        if save_enface_optic_nerve:

            if debug:
                print("Saving optic nerve labels")
            init_optic_nerve_labels = enface_onnx_seg_func(
                init_enface_data,
                onnx_path=EnfaceSegmentationType.OPTICNERVEHEAD.value,
                segmentation_type="ridge",
                label_val=6,
                use_cpu=True,
                DoG=True,
                blur=False,
            )
            on_labels_name = f"{name}_{en_face_ON_suffix}.npy"
            on_labels_name_path = output_dir/on_labels_name
            np.save(on_labels_name_path,init_optic_nerve_labels.astype(label_dtype.value))

        if save_enface_vessel:

            if debug:
                print("Saving vessel labels")
            init_vessel_labels = enface_onnx_seg_func(
                init_enface_data,
                onnx_path=EnfaceSegmentationType.VESSEL.value,
                segmentation_type="ridge",
                label_val=10,
                use_cpu=True,
                DoG=True,
                blur=True,
            )
            vessel_labels_name = f"{name}_{en_face_vessel_suffix}.npy"
            vessel_labels_name_path = output_dir/vessel_labels_name
            np.save(vessel_labels_name_path,save_enface_vessel.astype(label_dtype.value))

        if debug:
            print("Saving en face labels")       

        # TODO Reimplement later
        # combo_labels = init_optic_nerve_labels+init_ridge_labels+init_vessel_labels
        # combo_labels_name = f"{name}_{en_face_labels_suffix}.npy"
        # combo_labels_name_path = output_dir/combo_labels_name
        # np.save(combo_labels_name_path,combo_labels.astype(label_dtype.value))

        if save_pre_proc:
            preproc_name = f"{name}_{preproc_suffix}.npy"
            preproc_path = output_dir/preproc_name
            np.save(preproc_path,preproc_data.astype(img_dtype.value))

        if debug:
            print("En face labels saved")  


        if debug:
            print("Generating en face Data complete")
        
        #viewer.layers.remove(init_oct_data_layer)
        
        # print(type(enface_data))
        # print(enface_data)
        # print(enface_data.shape)
        # viewer.add_image(enface_data)
        # viewer.add_labels(ridge_labels)
        if show_in_napari:
            viewer.add_image(init_enface_data)
            viewer.add_labels(init_ridge_labels)
            viewer.add_labels(init_optic_nerve_labels)
            viewer.add_labels(init_vessel_labels)

    print("Batch Processing Complete.")

    if show_in_napari:
        viewer.show()
        napari.run()

    # o3d.visualization.draw_geometries([o3d_pcd])


# view_bscan_variants.changed.connect(print)
# app = QApplication([])
generate_enface_with_labels.show(run=True)
# app.exec_()
