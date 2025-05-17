"""
"""
import os
import gc
import napari
from enum import Enum
from tqdm import tqdm
from pathlib import Path
from magicgui import magic_factory

# Implement threading for this

class OutputFormat(Enum):
    """Enum for various output formats."""
    PROF = ".prof"
    PYTORCH = ".pt"
    MAT = ".mat"
    TIF = ".tif"

def _on_init(widget):
    """
    """
    optional = [
        "oct_avg",
        "bscans_to_avg",
        "mscans_to_avg",
        "oct_en_CLAHE",
        "oct_en_clahe_clip",
        "oct_en_log_corr",
        "oct_en_log_gain",
        "octa_en_CLAHE",
        "octa_en_clahe_clip",
        "octa_en_log_corr",
        "octa_en_log_gain",
        "octa_en_exp",
        "octa_en_n"
    ]
    
    #print(f"\n{dir(widget)}\n")
    
    for w in widget:
        if w.name in optional:
            w.visible = False
    
    widget.advanced.changed.connect(lambda val: show_hide_advanced(widget,val))

def show_hide_advanced(widget,val):
    """
    """
    optional = [
        "oct_avg",
        "bscans_to_avg",
        "mscans_to_avg",
        "oct_en_CLAHE",
        "oct_en_clahe_clip",
        "oct_en_log_corr",
        "oct_en_log_gain",
        "octa_en_CLAHE",
        "octa_en_clahe_clip",
        "octa_en_log_corr",
        "octa_en_log_gain",
        "octa_en_exp",
        "octa_en_n"
    ]

    if val == True:
        for w in widget:
            if w.name in optional:
                w.visible = True
    else:
        for w in widget:
            if w.name in optional:
                w.visible = False

@magic_factory(
        widget_init=_on_init,
        target_dir={"label": "Target Directory","mode":"d","tooltip": "Directory Containing files or subdirectories with OCT .prof files to process"},
)
def batch_proc_profs(
    target_dir:Path=Path(r"D:\JJ\Development\OCTA\Batch_Test"), #r"D:\Mani\To Segment\08851930-2023_08_23-13_32_03"), #r"D:\JJ\Development\OCTA\Batch_Test"), #(r"D:\Dir_to_process"),
    save_dir:Path=Path(r"D:\Dir_to_save_to"),
    out_type:OutputFormat=OutputFormat.PYTORCH,
    advanced:bool=False, 
    proc_struct:bool=True,
    segment:bool=True,
    proc_octa_var:bool=True,
    gen_enface:bool=True,
    gen_s_chor_en:bool=True,
    gen_octa_en:bool=True,
    gen_ret_en:bool=True,
    gen_chor_en:bool=True,
    gen_rem_en:bool=True,
    oct_avg:bool=True,
    ascan_corr:bool=True,
    bscans_to_avg:int=5,
    mscans_to_avg:int=3,
    oct_en_CLAHE:bool=True,
    oct_en_clahe_clip:float=2.5,
    oct_en_log_corr:bool=True,
    oct_en_log_gain:float=1,
    octa_en_CLAHE:bool=True,
    octa_en_clahe_clip:float=2.5,
    octa_en_log_corr:bool=True,
    octa_en_log_gain:float=1,
    octa_en_exp:bool=True,
    octa_en_n:float=1.5,
    save_selected:bool=False,
    open_napari:bool=True 
):
    """
    """

    viewer = napari.Viewer(show=False)

    # get list of .prof files
    prof_files = target_dir.glob('**/*.prof')

    #"""
    for file in tqdm(prof_files,desc="Processing OCTs"):
        print(f"\nProcessing file:\n{file}\n")

        process_octs(
            viewer=viewer,
            target_file=file,
            save_dir=save_dir,
            out_type=out_type,
            proc_struct=proc_struct,
            segment=segment,
            proc_octa_var=proc_octa_var,
            gen_enface=gen_enface,
            gen_s_chor_en=gen_s_chor_en,
            gen_octa_en=gen_octa_en,
            gen_ret_en=gen_ret_en,
            gen_chor_en=gen_chor_en,
            gen_rem_en=gen_rem_en,
            oct_avg=oct_avg,
            ascan_corr=ascan_corr,
            bscans_to_avg=bscans_to_avg,
            mscans_to_avg=mscans_to_avg,
            oct_en_CLAHE=oct_en_CLAHE,
            oct_en_clahe_clip=oct_en_clahe_clip,
            oct_en_log_corr=oct_en_log_corr,
            oct_en_log_gain=oct_en_log_gain,
            octa_en_CLAHE=octa_en_CLAHE,
            octa_en_clahe_clip=octa_en_clahe_clip,
            octa_en_log_corr=octa_en_log_corr,
            octa_en_log_gain=octa_en_log_gain,
            octa_en_exp=octa_en_exp,
            octa_en_n=octa_en_n,
            save_selected=save_selected,
            open_napari=open_napari,
        )

        gc.collect()
    #"""

    #gc.collect()

    print(f"OCT processing complete")
    if open_napari:
        viewer.show()
        #napari.run()

#run_napari()

def process_octs(
    viewer,
    target_file:Path=Path(r"D:\input_file.ext"),
    save_dir:Path=Path(r"D:\output_dir"),
    out_type:OutputFormat=OutputFormat.PYTORCH,
    proc_struct:bool=True,
    segment:bool=True,
    proc_octa_var:bool=True,
    gen_enface:bool=True,
    gen_s_chor_en:bool=True,
    gen_octa_en:bool=True,
    gen_ret_en:bool=True,
    gen_chor_en:bool=True,
    gen_rem_en:bool=True,
    oct_avg:bool=True,
    ascan_corr:bool=True,
    bscans_to_avg:int=5,
    mscans_to_avg:int=3,
    oct_en_CLAHE:bool=True,
    oct_en_clahe_clip:float=2.5,
    oct_en_log_corr:bool=True,
    oct_en_log_gain:float=1,
    octa_en_CLAHE:bool=True,
    octa_en_clahe_clip:float=2.5,
    octa_en_log_corr:bool=True,
    octa_en_log_gain:float=1,
    octa_en_exp:bool=True,
    octa_en_n:float=1.5,
    save_selected:bool=False,
    open_napari:bool=True 
):
    """
    """
    from napari_cool_tools_vol_proc._averaging_tools_funcs import average_bscans, average_per_bscan_pt
    from napari_cool_tools_oct_preproc._oct_preproc_utils_funcs import generate_enface, generate_octa_var
    from napari_cool_tools_registration._registration_tools_funcs import a_scan_correction_func2
    from napari_cool_tools_segmentation._segmentation_funcs import bscan_onnx_seg_func
    from napari_cool_tools_vol_proc._masking_tools_funcs import isolate_labeled_volume

    if save_selected:
        #os.makedirs(save_dir,exist_ok=True)
        pass

    viewer.open(target_file)
    print(target_file, "loaded")
    print(f"layer name is: {viewer.layers[0].name}")
    prof_layer = viewer.layers[-1]
    prof_data = prof_layer.data
    prof_name = prof_layer.name

    is_struc = is_octa = False

    if len(prof_data) > 1600:
        print(f"OCTA detected\n")
        is_octa = True
    else:
        print(f"Structural OCT detected\n")
        is_struc = True

    if is_octa:
        
        if proc_struct or segment or gen_enface or gen_octa_en or gen_ret_en or gen_chor_en or gen_rem_en or gen_s_chor_en:

            struc_name = f"{prof_name}_struc"

            print(f"Averaging M-scans\n")
            octa_struct = average_bscans(prof_data,scans_per_avg=mscans_to_avg)
            octa_base = octa_struct.copy()

            #viewer.add_image(octa_struct,name=f"{prof_name}_Mscan({mscans_to_avg})")

            if gen_enface:
                enface_name = f"{prof_name}_enface"
                print(f"Generating enface\n")
                gen_out = generate_enface(octa_struct,sin_correct=ascan_corr,exp=False,CLAHE=oct_en_CLAHE,clahe_clip=oct_en_clahe_clip,log_correct=oct_en_log_corr,log_gain=oct_en_log_gain,band_pass_filter=False)
                octa_en_data = next(gen_out)[0]

                
                if open_napari:
                    viewer.add_image(octa_en_data,name=enface_name)

            if ascan_corr:
                struc_name = f"{struc_name}_ASC"
                octa_struct = a_scan_correction_func2(octa_struct)
                   
            if oct_avg:
                print(f"Averaging B-scans\n")
                struc_name = f"{struc_name}_avg({bscans_to_avg})"
                octa_struct = average_per_bscan_pt(octa_struct,scans_per_avg=bscans_to_avg,trim=False,ensemble=True)
                octa_base = octa_struct.copy()            

            if proc_struct:
                if open_napari:
                    viewer.add_image(octa_struct,name=struc_name)

            if segment or gen_ret_en or gen_chor_en or gen_rem_en or gen_s_chor_en:        
                onnx_path = Path(r"../onnx_models/bscan/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856.onnx")
                segmentation = bscan_onnx_seg_func(octa_struct,onnx_path=onnx_path,use_cpu=False)[0][0] # change cpu to true for training!!

                if segment:
                    if open_napari:
                        viewer.add_labels(segmentation,name=f"{struc_name}_segmentation")

        if gen_s_chor_en:
            choroid = isolate_labeled_volume(octa_base,segmentation,label=2)
            chor_en = next(generate_enface(choroid,sin_correct=False,exp=octa_en_exp,n=octa_en_n,CLAHE=octa_en_CLAHE,clahe_clip=octa_en_clahe_clip,log_correct=octa_en_log_corr,log_gain=octa_en_log_gain,band_pass_filter=False))[0] #change sin_correct to true when you switch back to un ascan corrected volume for enface generation
            
            if open_napari:
                    viewer.add_image(choroid,name=f"{struc_name}_choroid")
            if open_napari:
                viewer.add_image(chor_en,name=f"{prof_name}_struc_chor_enface")

        if proc_octa_var or gen_octa_en or gen_ret_en or gen_chor_en or gen_rem_en:

            print(f"Generating OCTA variance data")
            #outputs = generate_octa(prof_data,enface_only=False,ascan_corr=ascan_corr,log_corr=True,clahe=True,log_gain=1,clahe_clip=2.5,octa_data_avg=5)
            outputs = generate_octa_var(prof_data,ascan_corr=True,w_wo_ascan=True)

            var = next(outputs)[0]
            var_asc = next(outputs)[0]

            var_name = f"{prof_name}"
            
            if proc_octa_var:
                if ascan_corr:
                    if open_napari:
                        var_name = f"{prof_name}_Var_ASC"
                        viewer.add_image(var_asc,name=var_name)
                else:
                    if open_napari:
                        var_name = f"{prof_name}_Var"
                        viewer.add_image(var,name=var_name)

            if gen_octa_en:
                octa_en = next(generate_enface(var,sin_correct=ascan_corr,exp=octa_en_exp,n=octa_en_n,CLAHE=octa_en_CLAHE,clahe_clip=octa_en_clahe_clip,log_correct=octa_en_log_corr,log_gain=octa_en_log_gain,band_pass_filter=False))[0]
                if open_napari:
                    viewer.add_image(octa_en,name=f"{prof_name}_OCTA_enface")

            if gen_ret_en:
                retina = isolate_labeled_volume(var_asc,segmentation,label=1)
                ret_en = next(generate_enface(retina,sin_correct=False,exp=octa_en_exp,n=octa_en_n,CLAHE=octa_en_CLAHE,clahe_clip=octa_en_clahe_clip,log_correct=octa_en_log_corr,log_gain=octa_en_log_gain,band_pass_filter=False))[0] #change sin_correct to true when you witch back to un ascan corrected volume for enface generation
                
                if proc_octa_var:
                    if open_napari:
                        viewer.add_image(retina,name=f"{var_name}_retina")
                if open_napari:
                    viewer.add_image(ret_en,name=f"{prof_name}_ret_enface")

            if gen_chor_en:
                choroid = isolate_labeled_volume(var_asc,segmentation,label=2)
                chor_en = next(generate_enface(choroid,sin_correct=False,exp=octa_en_exp,n=octa_en_n,CLAHE=octa_en_CLAHE,clahe_clip=octa_en_clahe_clip,log_correct=octa_en_log_corr,log_gain=octa_en_log_gain,band_pass_filter=False))[0] #change sin_correct to true when you witch back to un ascan corrected volume for enface generation
                
                if proc_octa_var:
                    if open_napari:
                        viewer.add_image(choroid,name=f"{var_name}_choroid")
                if open_napari:
                    viewer.add_image(chor_en,name=f"{prof_name}_chor_enface")

            if gen_rem_en:
                remainder = isolate_labeled_volume(var_asc,segmentation,label=0)
                rem_en = next(generate_enface(remainder,sin_correct=False,exp=octa_en_exp,n=octa_en_n,CLAHE=octa_en_CLAHE,clahe_clip=octa_en_clahe_clip,log_correct=octa_en_log_corr,log_gain=octa_en_log_gain,band_pass_filter=False))[0] #change sin_correct to true when you witch back to un ascan corrected volume for enface generation
                
                if proc_octa_var:
                    if open_napari:
                        viewer.add_image(remainder,name=f"{var_name}_remainder")
                if open_napari:
                    viewer.add_image(rem_en,name=f"{prof_name}_rem_enface")     

    else:
        pass

batch_proc_profs_gui = batch_proc_profs()
batch_proc_profs_gui.show(run=True)