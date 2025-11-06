@echo off

:start

pushd %~dp0

echo: &echo Opening napari using Pixi and UV. Please wait... &echo:
call pixi run napari

