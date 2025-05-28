"""
"""

import sys
import subprocess
import tomlkit
from typing import Literal,Tuple,List

from magicgui import magicgui


def main()->Tuple[str,str]:
    @magicgui(
        call_button="Install Napari",
    )
    def napari_install(
        version:Literal["development","production"] = "production",
        backend:Literal["cpu","cuda11","cuda12"] = "cpu"
        ):
        """
        Stacks .SLO files along new z axis.

        Parameters
        ----------
        SLO_dir: Directory Containing .SLO files to be converted
        output_dir: Directory to Store processed .SLO files if left empty will save to SLO_dir
        output_file_suffix: suffix to put between orginal file name and .SLO extension

        """
        features:List[str] = [backend]
        dev_status = ""
        if version == "development":
            dev_status = version
            features.append(dev_status)

        with open("pixi.toml","r") as f:
            pixi_config = tomlkit.load(f) 
        
        pixi_config["environments"]["default"] = {"features":features}
        print(f"Default feature config set to: {features}\n")

        with open('pixi.toml', 'w') as f:
            tomlkit.dump(pixi_config, f)

        subprocess.check_call( ['pixi', 'install']) 

        #if version == "development":
        #    settings = f"{backend}-dev"
        #    features:List[str] = ['-e', settings]
        #else:
        #    features:List[str] = ['-e', backend]
        #command:List[str] = ['pixi', 'install'] + features
        #subprocess.check_call([sys.executable, '-m', 'uv', 'sync', '--extra', backend])
        #subprocess.check_call(['uv', 'sync', '--extra', backend])
        #subprocess.check_call(command)
        print(napari_install)
        napari_install.close()

        return


    napari_install.show(run=True)

if __name__ == '__main__':
    main()