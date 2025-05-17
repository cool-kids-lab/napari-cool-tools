import pathlib
import xml.etree.ElementTree as ET
import os
import os.path as ospath
import subprocess
from magicgui import magicgui
from string import Template
from pathlib import Path
from napari.utils.notifications import show_info
from napari_cool_tools_io._prof_reader import prof_proc_meta

@magicgui(
    inputfolder={"label": "Input folder", "mode": "d"},
    outputfolder={"label": "Output folder", "mode": "d"},
    globalini={"label": "Global .ini file", "mode": "r"},
    initemplate={"label": "Template .ini file", "mode": "r"},
    batchOCTproc={"label": "OCTProcess.exe", "mode":"r"},
    call_button="Process folders",
)
def process_folders(
    inputfolder: pathlib.Path = Path("E:\\Choroidal Thickness Unp\\08877331\\8-16-2023"), 
    outputfolder: pathlib.Path = Path("D:\\JJ\\COOL_Lab\\Pipeline_Shakedown\\Spencer_Data_Testing"),
    globalini: pathlib.Path = Path("D:\\JJ\\Development\\COOL_Tools_plugin\config\\global.ini"), 
    initemplate: pathlib.Path = Path("D:\\JJ\\Development\\COOL_Tools_plugin\config\\template.ini"),
    batchOCTproc: pathlib.Path = Path("D:\\JJ\\Development\\OCTViewer_8_16_2023\\Release\\OCTProcess"),
    generateprofs: bool = True

    ):
    """
    """
    # do something with the folders
    print(f"input folder: {inputfolder}\noutput folder: {outputfolder}\n.ini template {initemplate}")
    
    # get .ini template
    t = Template(open(initemplate,"r"))
    
    # get list of .xml metafiles
    xml_meta_files = inputfolder.glob('**/*.xml')
    for file in xml_meta_files:
        print(f"file name: {file}")

        meta = prof_proc_meta(Path(file),".xml")

        print(meta)

        if meta is not None:
            h, w, d, bmscan, w_param, dtype, layer_type = meta

            if bmscan > 1:
                bidir = "false"
                bidir_a = "false"
            else:
                bidir = "true"
                bidir_a = "true"

            meta_params = {
                "width": w_param,
                "height": h,
                "frames": d,
                "bidir": bidir,
                "bidir_a": bidir_a,
                "bscan_width":w,
                "bmscan": bmscan
            }

            print(f"meta params {meta_params}")

            # open template and save changes out
            template = None
            filled_out = None
            with open(initemplate,"r") as template_file:
                template = template_file.read()
                #print(string)
                t = Template(template)
                filled_out = t.substitute(meta_params)
            #print(f"filled out template:\n{filled_out}")
            print(type(file),file.stem, file.parent, file.parent / (file.stem + ".ini"))
            
            ouput_file_path = file.parent / (file.stem + ".ini")
            with open(ouput_file_path, 'w') as output_file:
                output_file.write(filled_out)
    
    if generateprofs:
        # run command line tool
        result = subprocess.run (
            [batchOCTproc, "-i", inputfolder, "-o", outputfolder, "-g", globalini, "--prof", "--structure", "--preserve-dir", "--write-ini", "--start", "--exit"], 
            capture_output=False)
        print("Finished generateing .prof files.")
        
    print("Batch Processing Complete.")

#process_folders.changed.connect(print)
process_folders.show(run=True)