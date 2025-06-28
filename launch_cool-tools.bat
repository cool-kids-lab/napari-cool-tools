@echo off

:start

pushd %~dp0
echo: &echo Opening napari using Pixi and UV &echo:
echo: &echo Please keep this cmd window open for debugging purpose &echo:
pixi run napari
