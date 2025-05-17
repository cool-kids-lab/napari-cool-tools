@echo off

SET environment_name=cool-tools-env

:start
cls

pushd %~dp0
call dir
echo: &echo Confirm correct directory &echo:

call conda --version
echo: &echo step 1 Conda installation confirmed &echo:

::pause
::exit

call conda create -n %environment_name% python=3.9 -y
echo: &echo step 2 Conda environment installed &echo:

call conda activate %environment_name%
echo: &echo step 3 Conda environment activated &echo:

call conda env list
echo: &echo Conda envrionment activation confirmation &echo:


::pause
::exit

call pip install "numpy>1.23,<1.24"
echo: &echo step 4 Install specified numpy version &echo:

call pip install "napari[all]"
echo: &echo step 5 Install specified napari version &echo:

call pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
echo: &echo step 6 Install specified pytorch version &echo:

call pip install flatbuffers packaging protobuf
echo: &echo step 7 Install specified onnxruntime dependencies &echo:

call pip install onnxruntime-gpu
echo: &echo step 8 Install specified onnxruntime version &echo:

call pip install cupy-cuda12x
echo: &echo step 9 Install specified cupy version &echo:


call pip install kornia
echo: &echo step 10 Install specified kornia version &echo:

call pip install segmentation-models-pytorch
echo: &echo step 11 Install specified segmentation-models-pytorch version &echo:

call pip install matplotlib
echo: &echo step 12 Install specified matplotlib version &echo:

call pip install ruamel.yaml
echo: &echo step 13 Install specified ruamel.yaml version &echo:


call pip install -e ./napari-cool-tools-submenu-patch
echo: &echo step 14 install img-proc plugin &echo:
 
call pip install -e ./napari-cool-tools-img-proc
echo: &echo step 14 install img-proc plugin &echo:

call pip install -e ./napari-cool-tools-io
echo: &echo step 15 install io plugin &echo:

call pip install -e ./napari-cool-tools-oct-preproc
echo: &echo step 16 install oct-preproc plugin &echo:

call pip install -e ./napari-cool-tools-registration
echo: &echo step 17 install registration plugin &echo:

call pip install -e ./napari-cool-tools-segmentation
echo: &echo step 18 install segmentation plugin &echo:

call pip install -e ./napari-cool-tools-vol-proc
echo: &echo step 19 install vol-proc plugin &echo:


::pause
::exit

echo: &echo step 15 creating shortcut on Desktop &echo:

set SCRIPT="%TEMP%\%RANDOM%-%RANDOM%-%RANDOM%-%RANDOM%.vbs"
echo Set oWS = WScript.CreateObject("WScript.Shell") >> %SCRIPT%
echo sLinkFile = "%USERPROFILE%\Desktop\Napari Cool Tools.lnk" >> %SCRIPT%
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> %SCRIPT%
echo oLink.TargetPath = "%CD%\launch_cool-tools_conda.bat" >> %SCRIPT%
echo oLink.IconLocation = "%CD%\napari.ico" >> %SCRIPT%
echo oLink.Save >> %SCRIPT%
cscript /nologo %SCRIPT%
del "%SCRIPT%"

exit