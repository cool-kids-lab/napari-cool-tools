@echo off

:start

pushd %~dp0
call dir
echo: &echo Confirm correct directory &echo:

call pixi run napari
echo: &echo run napari using Pixi and UV &echo: