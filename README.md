# napari-cool-tools
 Napari Library Developed by COOL LAB group at Casey Eye Institute, Oregon Health and Science University

# Minimum System Requirements
 - Windows 10 or later
 - CUDA 12.4 (NVIDIA GPU with CUDA Capabilities
 - Conda 24.0 or later
 - Napari 0.5.6 (will be installed with the installer.bat)

 Note: This library hasn't been tested in OS other than Windows 10 or later. Although it is possible to install it manually on Linux or MacOS, some feature may not be running properly.

# Installation Instruction
 ## 1. Make  sure CUDA Libray is installed (download it here [CUDA](https://developer.nvidia.com/cuda-12-4-0-download-archive))
   Check your cuda installation by run `nvcc` in the command prompt.
 ## 2. Make  sure Conda or Miniconda is installed (download it here [CONDA](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe))
   Check your conda installation by run `conda --version` in the command prompt.
 ## 3. Double Click the `installer.bat` file
   After the installation is finished. A shortcut is created on the Desktop folder. Two options are available to run the napari.
   1. Double click the `Napari Cool Tools` shortcut in Desktop folder.
      or
   2. Open powershell or command promp, and activate the environment by running `conda activate cool-tools-env`, and then run `napari`
