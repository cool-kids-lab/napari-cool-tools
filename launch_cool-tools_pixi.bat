@echo off

:start

pushd %~dp0

echo: &echo running napari using Pixi and UV &echo:
call pixi run napari

