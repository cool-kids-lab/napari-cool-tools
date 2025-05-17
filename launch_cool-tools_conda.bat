@echo off
SET mypath=%~dp0
call conda --version
call conda activate cool-tools-env
call conda env list
echo %mypath:~,-1%
call napari --version
call napari