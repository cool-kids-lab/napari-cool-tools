import os 
from pathlib import Path
from magicgui import magic_factory

@magic_factory(
        #widget_init=_on_init,
        target_file={"label": "Target Directory","mode":"r","tooltip": "Directory Containing files or subdirectories with OCT .prof files to process"},
)
def parse_ini_file(
    target_file:Path=Path(r"E:\_UWF_OCTA_DESKTOP_JJ\2024-09-19T14_54_27_318573Z-07_00_UNP.ini")
):
    #print(target_file)
    section = ""
    sec_content = ""
    data = {"section":"", "content":""}
    settings = []

    width_param, height, width, depth, bmscan = (
            None,
            None,
            None,
            None,
            None,
        )
    
    def ini_proc_word(line,target_str):
        """"""
        words = line.split("=")
        index = words.index(target_str)
        if index + 1 < len(words):
            val = int(words[index + 1])
            return val
        else:
            print("ERROR in ini_proc_word function")
            return None

    with open(target_file) as file:
        for i,line in enumerate(file):
            if "[" not in line:
                data["content"] = f"{data['content']}{line}"

                if data["section"] == "General" and "WIDTH=" in line:
                    width_param = ini_proc_word(line,"WIDTH")
                if data["section"] == "General" and "HEIGHT=" in line:
                    height = ini_proc_word(line,"HEIGHT")
                if data["section"] == "General" and "FRAMES=" in line:
                    depth = ini_proc_word(line,"FRAMES")
                if data["section"] == "OCT" and "BScanWidth=" in line:
                    width = ini_proc_word(line,"BScanWidth")
                if data["section"] == "OCTA" and "BMScan=" in line:
                    bmscan = ini_proc_word(line,"BMScan")
            else:
                if i != 0:
                    settings.append(data)
                data = {"section":"", "content":""}
                data["section"] = line.replace("[","").replace("]","").replace("\n","")
                settings
        settings.append(data)
    
    dtype = None
    layer_type = None

    print(settings)
    print(
        height,
        width,
        depth,
        bmscan,
        width_param,
        dtype,
        layer_type,
        )            

batch_proc_profs_gui = parse_ini_file()
batch_proc_profs_gui.show(run=True)