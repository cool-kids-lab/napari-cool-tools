import pathlib
import xml.etree.ElementTree as ET
import os
import os.path as ospath
import subprocess
import napari
from enum import Enum
from tqdm import tqdm
#import napari_cool_tools_io
from magicgui import magicgui, magic_factory
from pathlib import Path
from napari.qt.threading import thread_worker
from napari_cool_tools_io import _prof_reader, _prof_writer
from napari_cool_tools_oct_preproc import _oct_preproc_utils_funcs #generate_enface_image

class OutputFormat(Enum):
    """Enum for various output formats."""
    PROF = ".prof"
    PYTORCH = ".pt"
    MAT = ".mat"
    TIF = ".tif"

def _on_init(widget):
    """"""
    widget.advanced.changed.connect(lambda val: process_oct_dir_change(widget,val))


def process_oct_dir_change(widget,val):
    """"""
    optional = ["oct_avg",
                "bscans_to_avg",
                "trim",
                "gen_enface",
                "save_struc",
                "mscans_to_avg",
                "gen_octa_en",
                "gen_ret_en",
                "gen_chor_en",
                "save_octa_var"
                ]
    
    oct_struct_op = [
        "bscans_to_avg",
        "trim",
        "gen_enface",
    ]

    octa_op = [
        "save_struc",
        "mscans_to_avg",
        "gen_octa_en",
        "gen_ret_en",
        "gen_chor_en",
        "save_octa_var"
    ]

    for k in optional:
        if k in widget.asdict().keys():
            widget[k].visible = False
        else:
            pass
    
    if widget["proc_struc"] == True:
        for k in oct_struct_op:
            if k in widget.asdict().keys():
                widget[k].visible = True

    if widget["proc_octa"] == True:
        for k in octa_op:
            if k in widget.asdict().keys():
                widget[k].visible = True

def process_octs_struc(widget,val):
    """"""
    optional = ["oct_avg","bscans_to_avg","trim","save_struc","mscans_to_avg"]

    for k in optional:
        if k in widget.asdict().keys():
            widget[k].visible = False
        else:
            pass
    
    if val == True:
        for k in optional:
            if k in widget.asdict().keys():
                widget[k].visible = True

def process_octs_type(widget,val):
    """"""
    optional = ['key','img','lbls']

    for k in optional:
        if k in widget.asdict().keys():
            widget[k].visible = False
        else:
            pass

    if val == OutputFormat.Data:
        pass
    elif val == OutputFormat.Dict:
        widget.key.visible = True

    elif val == OutputFormat.Img_Lbl_Pr:
        widget.file_name.visible = True
        widget.img_key.visible = True
        widget.lbl_key.visible = True
        widget.img.visible = True
        widget.lbls.visible = True

{"label": "", "tooltip": ""}

"""
@magic_factory(
#@magicgui(
    widget_init=_on_init,
    target_dir={"label": "Input folder", "mode": "d","tooltip": "Folder containing OCTs to be processed"},
    save_dir={"label": "File name", "mode":"d", "tooltip": "Folder to save processed files to"},
    out_type={"label": "Output Format"},
    advanced={"label": "Display Advanced Options"},
    #proc_struc={"label": "Proc struc OCT", "tooltip": "Process Structural OCTs"},
    gen_enface={"label": "Generate Enface"},
    #proc_octa={"label": "Proc OCTAs", "tooltip": "Process OCTAs"},
    save_struct={"label": "Save Struct","tooltip": "Save structural OCT data"},
    ascan_corr={"label": "Ascan Correction","tooltip": "Ascan correction"},
    oct_avg={"label": "Avg Bscan Toggle"},
    bscans_to_avg={"label": "Bscans to Avg"},
    trim={"label":"Trim unaveraged bescans"},
    mscans_to_avg={"label": "Mscans to Avg","tooltip": "Number of Mscans to average"},
    gen_octa_en={"label": "Gen OCTA enface","tooltip": "Generate OCTA"},
    gen_ret_en={"label": "Gen Retina enface","tooltip": "Generate retina enface"},
    gen_chor_en={"label": "Gen Choroid enface","tooltip": "Generate choroid enface"},
    gen_rem_en={"label": "Gen Remainder enface","tooltip": "Generate remainder enface"},
    save_octa_var={"label": "Save OCTA var","tooltip": "Save OCTA variance data"},
    open_napari={"label": "Open Napari","tooltip": "View processed files in napari"},    
    call_button="Process OCTs",
)
"""

@magic_factory()
def process_oct_dir(
    target_dir:Path=Path(r"D:\JJ\Development\OCTA\Batch_Test"), #(r"D:\Dir_to_process"),
    save_dir:Path=Path(r"D:\Dir_to_save_to"),
    out_type:OutputFormat=OutputFormat.PYTORCH,
    advanced:bool=False,
    gen_enface:bool=True,
    save_struct:bool=True,
    ascan_corr:bool=True,
    oct_avg:bool=True,
    bscans_to_avg:int=5,
    trim:bool=False,
    mscans_to_avg:int=3,
    gen_octa_en:bool=True,
    gen_ret_en:bool=True,
    gen_chor_en:bool=True,
    gen_rem_en:bool=True,
    save_octa_var:bool=False,
    open_napari:bool=False 
):
    """
    """
    # create headless napari viewer

    viewer = napari.Viewer(show=False)

    print(f"Do stuff")

    viewer.show()

    """
    # get list of .prof files
    prof_files = target_dir.glob('**/*.prof')

    for file in tqdm(prof_files,desc="Processing OCTs"):
        print(f"\nProcessing file:\n{file}\n")

        #process_octs2(viewer,file)

        process_octs(
            viewer=viewer,
            target_file=file,
            save_dir=save_dir,
            out_type=out_type,
            advanced=advanced,
            gen_enface=gen_enface,
            save_struct=save_struct,
            oct_avg=oct_avg,
            bscans_to_avg=bscans_to_avg,
            trim=trim,
            ascan_corr=ascan_corr,
            mscans_to_avg=mscans_to_avg,
            gen_octa_en=gen_octa_en,
            gen_ret_en=gen_ret_en,
            gen_chor_en=gen_chor_en,
            gen_rem_en=gen_rem_en,
            save_octa_var=save_octa_var,
            open_napari=open_napari,
        )

    print(f"Processing Complete\n")
    if open_napari:
        viewer.show()
        napari.run()
    
    """

def process_octs2(viewer,target_file:Path):
    """
    """
    viewer.open(target_file)
    print(target_file, "loaded")
    print(f"layer name is: {viewer.layers[-1].name}")

def process_octs(
    viewer,
    target_file:Path=Path(r"D:\input_file.ext"),
    save_dir:Path=Path(r"D:\output_dir"),
    out_type:OutputFormat=OutputFormat.PYTORCH,
    advanced:bool=False,
    #proc_struc:bool=True,
    gen_enface:bool=True,
    #proc_octa:bool=True,
    save_struct:bool=True,
    ascan_corr:bool=True,
    oct_avg:bool=True,
    bscans_to_avg:int=5,
    trim:bool=False,
    mscans_to_avg:int=3,
    gen_octa_en:bool=True,
    gen_ret_en:bool=True,
    gen_chor_en:bool=True,
    gen_rem_en:bool=True,
    save_octa_var:bool=False,
    open_napari:bool=False    
):
    """
    """
    from napari_cool_tools_vol_proc._averaging_tools_funcs import average_bscans, average_per_bscan_pt
    from napari_cool_tools_oct_preproc._oct_preproc_utils_funcs import generate_enface, generate_octa
    from napari_cool_tools_registration._registration_tools_funcs import a_scan_correction_func2
    from napari_cool_tools_segmentation._segmentation_funcs import bscan_onnx_seg_func
    from napari_cool_tools_vol_proc._masking_tools_funcs import isolate_labeled_volume

    viewer.open(target_file)
    print(target_file, "loaded")
    print(f"layer name is: {viewer.layers[0].name}")
    prof_layer = viewer.layers[-1]
    prof_data = prof_layer.data
    prof_name = prof_layer.name

    proc_struc = proc_octa = False

    if len(prof_data) > 1600:
        print(f"OCTA detected\n")
        proc_octa = True
    else:
        print(f"Structural OCT detected\n")
        proc_struc = True


    if proc_octa:
        
        if save_struct or gen_enface:
            print(f"Averaging M-scans\n")
            octa_struct = average_bscans(prof_data,scans_per_avg=mscans_to_avg)

            viewer.add_image(octa_struct,name=f"{prof_name}_Mscan({mscans_to_avg})")

            if gen_enface:
                enface_name = f"{prof_name}_enface"
                print(f"Generating enface\n")
                gen_out = generate_enface(octa_struct,sin_correct=True,exp=False,CLAHE=True,clahe_clip=2.5,log_correct=True,log_gain=1,band_pass_filter=False)
                octa_en_data = next(gen_out)[0]

                viewer.add_image(octa_en_data,name=enface_name)

            if save_struct:
                struc_name = f"{prof_name}_struc"
                
                if oct_avg:
                    print(f"Averaging B-scans\n")
                    struc_name = f"{struc_name}_avg({bscans_to_avg})"
                    octa_struct = average_per_bscan_pt(octa_struct,scans_per_avg=bscans_to_avg,trim=trim,ensemble=True)

                if ascan_corr:
                    struc_name = f"{struc_name}_ASC"
                    octa_struct = a_scan_correction_func2(octa_struct)


                viewer.add_image(octa_struct,name=struc_name)

        if save_octa_var or gen_octa_en or gen_ret_en or gen_chor_en or gen_rem_en:

            print(f"Generating OCTA variance data")
            outputs = generate_octa(prof_data,enface_only=False,ascan_corr=True,log_corr=True,clahe=True,log_gain=1,clahe_clip=2.5,octa_data_avg=5)

            var = next(outputs)[0]
            var_asc = next(outputs)[0]
            for output in outputs:
                viewer.add_image(output[0],name=f"{prof_name}_{output[1]}")

            onnx_path = Path(r"../onnx_models/bscan/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856.onnx")
            segmentation = bscan_onnx_seg_func(octa_struct,onnx_path=onnx_path,use_cpu=False)[0][0] # change this!! to true for training
            #print(f"segmentation:\n{segmentation}\n")
            viewer.add_labels(segmentation,name=f"{prof_name}_segmentation")

            if gen_ret_en:
                retina = isolate_labeled_volume(var_asc,segmentation,label=1)
                ret_en = next(generate_enface(retina,sin_correct=False,exp=True,n=1.5,CLAHE=True,clahe_clip=2.5,log_correct=True,log_gain=1,band_pass_filter=False))[0] #change sin_correct to true when you witch back to un ascan corrected volume for enface generation
                viewer.add_image(retina,name=f"{prof_name}_retina")
                viewer.add_image(ret_en,name=f"{prof_name}_ret_enface")

            if gen_chor_en:
                choroid = isolate_labeled_volume(var_asc,segmentation,label=2)
                chor_en = next(generate_enface(choroid,sin_correct=False,exp=True,n=1.5,CLAHE=True,clahe_clip=2.5,log_correct=True,log_gain=1,band_pass_filter=False))[0] #change sin_correct to true when you witch back to un ascan corrected volume for enface generation
                viewer.add_image(choroid,name=f"{prof_name}_choroid")
                viewer.add_image(chor_en,name=f"{prof_name}_chor_enface")

            if gen_rem_en:
                remainder = isolate_labeled_volume(var_asc,segmentation,label=0)
                rem_en = next(generate_enface(remainder,sin_correct=False,exp=True,n=1.5,CLAHE=True,clahe_clip=2.5,log_correct=True,log_gain=1,band_pass_filter=False))[0] #change sin_correct to true when you witch back to un ascan corrected volume for enface generation
                viewer.add_image(remainder,name=f"{prof_name}_remainder")
                viewer.add_image(rem_en,name=f"{prof_name}_rem_enface")     

    else:
        pass

process_oct_gui = process_oct_dir()
process_oct_gui.show(run=True)